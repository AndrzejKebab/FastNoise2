#include "Simplex.h"
#include "Utils.inl"

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<Simplex, SIMD> final : public virtual Simplex, public DispatchClass<VariableRange<Seeded<ScalableGenerator>>, SIMD>
{
    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y );

        constexpr double kRoot3 = 1.7320508075688772935274463415059;
        constexpr double kSkew2 = 1.0 / ( kRoot3 + 1.0 );
        constexpr double kUnskew2 = -1.0 / ( kRoot3 + 3.0 );
        constexpr double kFalloffRadiusSquared = 0.5;

        float32v skewDelta = float32v( kSkew2 ) * ( x + y );
        float32v xSkewed = x + skewDelta;
        float32v ySkewed = y + skewDelta;

        float32v xSkewedBase = FS::Floor( xSkewed );
        float32v ySkewedBase = FS::Floor( ySkewed );
        float32v dxSkewed = xSkewed - xSkewedBase;
        float32v dySkewed = ySkewed - ySkewedBase;

        int32v xPrimedBase = FS::Convert<int32_t>( xSkewedBase ) * int32v( Primes::X );
        int32v yPrimedBase = FS::Convert<int32_t>( ySkewedBase ) * int32v( Primes::Y );

        mask32v xGreaterEqualY = dxSkewed >= dySkewed;
        
        float32v unskewDelta = float32v( kUnskew2 ) * ( dxSkewed + dySkewed );
        float32v dx0 = dxSkewed + unskewDelta;
        float32v dy0 = dySkewed + unskewDelta;
        
        float32v dx1 = FS::MaskedIncrement( ~xGreaterEqualY, dx0 ) - float32v( kUnskew2 + 1 );
        float32v dy1 = FS::MaskedIncrement( xGreaterEqualY, dy0 ) - float32v( kUnskew2 + 1 );
        float32v dx2 = dx0 - float32v( kUnskew2 * 2 + 1 );
        float32v dy2 = dy0 - float32v( kUnskew2 * 2 + 1 );

        float32v falloff0 = FS::FNMulAdd( dx0, dx0, FS::FNMulAdd( dy0, dy0, float32v( kFalloffRadiusSquared ) ) );
        float32v falloff1 = FS::FNMulAdd( dx1, dx1, FS::FNMulAdd( dy1, dy1, float32v( kFalloffRadiusSquared ) ) );
        float32v falloff2 = falloff0 + FS::FMulAdd( unskewDelta,
            float32v( -4.0 * ( kRoot3 + 2.0 ) / ( kRoot3 + 3.0 ) ),
            float32v( -2.0 / 3.0 ) );

        falloff0 = FS::Max( falloff0, float32v( 0 ) );
        falloff1 = FS::Max( falloff1, float32v( 0 ) );
        falloff2 = FS::Max( falloff2, float32v( 0 ) );

        falloff0 *= falloff0; falloff0 *= falloff0;
        falloff1 *= falloff1; falloff1 *= falloff1;
        falloff2 *= falloff2; falloff2 *= falloff2;

        float32v gradientRampValue0 = GetGradientDotPerlin( HashPrimes( seed, xPrimedBase, yPrimedBase ), dx0, dy0 );
        float32v gradientRampValue1 = GetGradientDotPerlin( HashPrimes( seed, FS::MaskedAdd( xGreaterEqualY, xPrimedBase, int32v( Primes::X ) ), FS::InvMaskedAdd( xGreaterEqualY, yPrimedBase, int32v( Primes::Y ) ) ), dx1, dy1 );
        float32v gradientRampValue2 = GetGradientDotPerlin( HashPrimes( seed, xPrimedBase + int32v( Primes::X ), yPrimedBase + int32v( Primes::Y ) ), dx2, dy2 );

        float32v value = FS::FMulAdd( gradientRampValue0, falloff0, FS::FMulAdd( gradientRampValue1, falloff1, gradientRampValue2 * falloff2 ) );

        constexpr float kBounding = 0.0261208079755306243896484375f;

        return this->ScaleOutput( value, -kBounding, kBounding );
    }

    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y, z );

        constexpr double kSkew3 = 1.0 / 3.0;
        constexpr double kReflectUnskew3 = -1.0 / 2.0;
        constexpr double kFalloffRadiusSquared = 0.6;

        float32v skewDelta = float32v( kSkew3 ) * ( x + y + z );
        float32v xSkewed = x + skewDelta;
        float32v ySkewed = y + skewDelta;
        float32v zSkewed = z + skewDelta;

        float32v xSkewedBase = FS::Floor( xSkewed );
        float32v ySkewedBase = FS::Floor( ySkewed );
        float32v zSkewedBase = FS::Floor( zSkewed );
        float32v dxSkewed = xSkewed - xSkewedBase;
        float32v dySkewed = ySkewed - ySkewedBase;
        float32v dzSkewed = zSkewed - zSkewedBase;

        int32v xPrimedBase = FS::Convert<int32_t>( xSkewedBase ) * int32v( Primes::X );
        int32v yPrimedBase = FS::Convert<int32_t>( ySkewedBase ) * int32v( Primes::Y );
        int32v zPrimedBase = FS::Convert<int32_t>( zSkewedBase ) * int32v( Primes::Z );

        mask32v xGreaterEqualY = dxSkewed >= dySkewed;
        mask32v yGreaterEqualZ = dySkewed >= dzSkewed;
        mask32v xGreaterEqualZ = dxSkewed >= dzSkewed;

        float32v unskewDelta = float32v( kReflectUnskew3 ) * ( dxSkewed + dySkewed + dzSkewed );
        float32v dx0 = dxSkewed + unskewDelta;
        float32v dy0 = dySkewed + unskewDelta;
        float32v dz0 = dzSkewed + unskewDelta;

        mask32v maskX1 = xGreaterEqualY & xGreaterEqualZ;
        mask32v maskY1 = FS::BitwiseAndNot( yGreaterEqualZ, xGreaterEqualY );
        mask32v maskZ1 = xGreaterEqualZ | yGreaterEqualZ; // Inv masked

        mask32v nMaskX2 = xGreaterEqualY | xGreaterEqualZ; // Inv masked
        mask32v nMaskY2 = FS::BitwiseAndNot( xGreaterEqualY, yGreaterEqualZ );
        mask32v nMaskZ2 = xGreaterEqualZ & yGreaterEqualZ;

        float32v dx3 = dx0 - float32v( kReflectUnskew3 * 3 + 1 );
        float32v dy3 = dy0 - float32v( kReflectUnskew3 * 3 + 1 );
        float32v dz3 = dz0 - float32v( kReflectUnskew3 * 3 + 1 );
        float32v dx1 = FS::MaskedSub( maskX1, dx3, float32v( 1 ) ); // kReflectUnskew3 * 3 + 1 = kReflectUnskew3, so dx0 - kReflectUnskew3 = dx3
        float32v dy1 = FS::MaskedSub( maskY1, dy3, float32v( 1 ) );
        float32v dz1 = FS::InvMaskedSub( maskZ1, dz3, float32v( 1 ) );
        float32v dx2 = FS::MaskedIncrement( ~nMaskX2, dx0 ); // kReflectUnskew3 * 2 - 1 = 0, so dx0 + ( kReflectUnskew3 * 2 - 1 ) = dx0
        float32v dy2 = FS::MaskedIncrement( nMaskY2, dy0 );
        float32v dz2 = FS::MaskedIncrement( nMaskZ2, dz0 );

        float32v falloff0 = FS::FNMulAdd( dz0, dz0, FS::FNMulAdd( dy0, dy0, FS::FNMulAdd( dx0, dx0, float32v( kFalloffRadiusSquared ) ) ) );
        float32v falloff1 = FS::FNMulAdd( dz1, dz1, FS::FNMulAdd( dy1, dy1, FS::FNMulAdd( dx1, dx1, float32v( kFalloffRadiusSquared ) ) ) );
        float32v falloff2 = FS::FNMulAdd( dz2, dz2, FS::FNMulAdd( dy2, dy2, FS::FNMulAdd( dx2, dx2, float32v( kFalloffRadiusSquared ) ) ) );
        float32v falloff3 = falloff0 - ( unskewDelta + float32v( 3.0 / 4.0 ) );

        falloff0 = FS::Max( falloff0, float32v( 0 ) );
        falloff1 = FS::Max( falloff1, float32v( 0 ) );
        falloff2 = FS::Max( falloff2, float32v( 0 ) );
        falloff3 = FS::Max( falloff3, float32v( 0 ) );

        falloff0 *= falloff0; falloff0 *= falloff0;
        falloff1 *= falloff1; falloff1 *= falloff1;
        falloff2 *= falloff2; falloff2 *= falloff2;
        falloff3 *= falloff3; falloff3 *= falloff3;

        float32v gradientRampValue0 = GetGradientDotCommon( HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimedBase ), dx0, dy0, dz0 );
        float32v gradientRampValue1 = GetGradientDotCommon( HashPrimes( seed, FS::MaskedAdd( maskX1, xPrimedBase, int32v( Primes::X ) ), FS::MaskedAdd( maskY1, yPrimedBase, int32v( Primes::Y ) ), FS::InvMaskedAdd( maskZ1, zPrimedBase, int32v( Primes::Z ) ) ), dx1, dy1, dz1 );
        float32v gradientRampValue2 = GetGradientDotCommon( HashPrimes( seed, FS::MaskedAdd( nMaskX2, xPrimedBase, int32v( Primes::X ) ), FS::InvMaskedAdd( nMaskY2, yPrimedBase, int32v( Primes::Y ) ), FS::InvMaskedAdd( nMaskZ2, zPrimedBase, int32v( Primes::Z ) ) ), dx2, dy2, dz2 );
        float32v gradientRampValue3 = GetGradientDotCommon( HashPrimes( seed, xPrimedBase + int32v( Primes::X ), yPrimedBase + int32v( Primes::Y ), zPrimedBase + int32v( Primes::Z ) ), dx3, dy3, dz3 );

        float32v value = FS::FMulAdd( gradientRampValue3, falloff3, FS::FMulAdd( gradientRampValue2, falloff2, FS::FMulAdd( gradientRampValue1, falloff1, gradientRampValue0 * falloff0 ) ) );

        constexpr float kBounding = 0.030586399137973785400390625f;

        return this->ScaleOutput( value, -kBounding, kBounding );
    }

    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z, float32v w ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y, z, w );

        constexpr double kRoot5 = 2.2360679774997896964091736687313;
        constexpr double kSkew4 = 1.0 / ( kRoot5 + 1.0 );
        constexpr double kUnskew4 = -1.0 / ( kRoot5 + 5.0 );
        constexpr double kFalloffRadiusSquared = 0.6;

        float32v skewDelta = float32v( kSkew4 ) * ( x + y + z + w );
        float32v xSkewed = x + skewDelta;
        float32v ySkewed = y + skewDelta;
        float32v zSkewed = z + skewDelta;
        float32v wSkewed = w + skewDelta;

        float32v xSkewedBase = FS::Floor( xSkewed );
        float32v ySkewedBase = FS::Floor( ySkewed );
        float32v zSkewedBase = FS::Floor( zSkewed );
        float32v wSkewedBase = FS::Floor( wSkewed );
        float32v dxSkewed = xSkewed - xSkewedBase;
        float32v dySkewed = ySkewed - ySkewedBase;
        float32v dzSkewed = zSkewed - zSkewedBase;
        float32v dwSkewed = wSkewed - wSkewedBase;

        int32v xPrimedBase = FS::Convert<int32_t>( xSkewedBase ) * int32v( Primes::X );
        int32v yPrimedBase = FS::Convert<int32_t>( ySkewedBase ) * int32v( Primes::Y );
        int32v zPrimedBase = FS::Convert<int32_t>( zSkewedBase ) * int32v( Primes::Z );
        int32v wPrimedBase = FS::Convert<int32_t>( wSkewedBase ) * int32v( Primes::W );

        float32v unskewDelta = float32v( kUnskew4 ) * ( dxSkewed + dySkewed + dzSkewed + dwSkewed );
        float32v dx0 = dxSkewed + unskewDelta;
        float32v dy0 = dySkewed + unskewDelta;
        float32v dz0 = dzSkewed + unskewDelta;
        float32v dw0 = dwSkewed + unskewDelta;

        int32v rankX( 0 );
        int32v rankY( 0 );
        int32v rankZ( 0 );
        int32v rankW( 0 );

        mask32v xGreaterEqualY = dx0 >= dy0;
        rankX = FS::MaskedIncrement(  xGreaterEqualY, rankX );
        rankY = FS::MaskedIncrement( ~xGreaterEqualY, rankY );

        mask32v xGreaterEqualZ = dx0 >= dz0;
        rankX = FS::MaskedIncrement(  xGreaterEqualZ, rankX );
        rankZ = FS::MaskedIncrement( ~xGreaterEqualZ, rankZ );

        mask32v xGreaterEqualW = dx0 >= dw0;
        rankX = FS::MaskedIncrement(  xGreaterEqualW, rankX );
        rankW = FS::MaskedIncrement( ~xGreaterEqualW, rankW );

        mask32v yGreaterEqualZ = dy0 >= dz0;
        rankY = FS::MaskedIncrement(  yGreaterEqualZ, rankY );
        rankZ = FS::MaskedIncrement( ~yGreaterEqualZ, rankZ );

        mask32v yGreaterEqualW = dy0 >= dw0;
        rankY = FS::MaskedIncrement(  yGreaterEqualW, rankY );
        rankW = FS::MaskedIncrement( ~yGreaterEqualW, rankW );

        mask32v zGreaterEqualW = dz0 >= dw0;
        rankZ = FS::MaskedIncrement(  zGreaterEqualW, rankZ );
        rankW = FS::MaskedIncrement( ~zGreaterEqualW, rankW );

        mask32v maskX1 = rankX > int32v( 2 );
        mask32v maskY1 = rankY > int32v( 2 );
        mask32v maskZ1 = rankZ > int32v( 2 );
        mask32v maskW1 = rankW > int32v( 2 );

        mask32v maskX2 = rankX > int32v( 1 );
        mask32v maskY2 = rankY > int32v( 1 );
        mask32v maskZ2 = rankZ > int32v( 1 );
        mask32v maskW2 = rankW > int32v( 1 );

        mask32v maskX3 = rankX > int32v( 0 );
        mask32v maskY3 = rankY > int32v( 0 );
        mask32v maskZ3 = rankZ > int32v( 0 );
        mask32v maskW3 = rankW > int32v( 0 );

        float32v dx1 = FS::MaskedSub( maskX1, dx0, float32v( 1 ) ) - float32v( kUnskew4 );
        float32v dy1 = FS::MaskedSub( maskY1, dy0, float32v( 1 ) ) - float32v( kUnskew4 );
        float32v dz1 = FS::MaskedSub( maskZ1, dz0, float32v( 1 ) ) - float32v( kUnskew4 );
        float32v dw1 = FS::MaskedSub( maskW1, dw0, float32v( 1 ) ) - float32v( kUnskew4 );
        float32v dx2 = FS::MaskedSub( maskX2, dx0, float32v( 1 ) ) - float32v( kUnskew4 * 2 );
        float32v dy2 = FS::MaskedSub( maskY2, dy0, float32v( 1 ) ) - float32v( kUnskew4 * 2 );
        float32v dz2 = FS::MaskedSub( maskZ2, dz0, float32v( 1 ) ) - float32v( kUnskew4 * 2 );
        float32v dw2 = FS::MaskedSub( maskW2, dw0, float32v( 1 ) ) - float32v( kUnskew4 * 2 );
        float32v dx3 = FS::MaskedSub( maskX3, dx0, float32v( 1 ) ) - float32v( kUnskew4 * 3 );
        float32v dy3 = FS::MaskedSub( maskY3, dy0, float32v( 1 ) ) - float32v( kUnskew4 * 3 );
        float32v dz3 = FS::MaskedSub( maskZ3, dz0, float32v( 1 ) ) - float32v( kUnskew4 * 3 );
        float32v dw3 = FS::MaskedSub( maskW3, dw0, float32v( 1 ) ) - float32v( kUnskew4 * 3 );
        float32v dx4 = dx0 - float32v( kUnskew4 * 4 + 1 );
        float32v dy4 = dy0 - float32v( kUnskew4 * 4 + 1 );
        float32v dz4 = dz0 - float32v( kUnskew4 * 4 + 1 );
        float32v dw4 = dw0 - float32v( kUnskew4 * 4 + 1 );

        float32v falloff0 = FS::FNMulAdd( dw0, dw0, FS::FNMulAdd( dz0, dz0, FS::FNMulAdd( dy0, dy0, FS::FNMulAdd( dx0, dx0, float32v( kFalloffRadiusSquared ) ) ) ) );
        float32v falloff1 = FS::FNMulAdd( dw1, dw1, FS::FNMulAdd( dz1, dz1, FS::FNMulAdd( dy1, dy1, FS::FNMulAdd( dx1, dx1, float32v( kFalloffRadiusSquared ) ) ) ) );
        float32v falloff2 = FS::FNMulAdd( dw2, dw2, FS::FNMulAdd( dz2, dz2, FS::FNMulAdd( dy2, dy2, FS::FNMulAdd( dx2, dx2, float32v( kFalloffRadiusSquared ) ) ) ) );
        float32v falloff3 = FS::FNMulAdd( dw3, dw3, FS::FNMulAdd( dz3, dz3, FS::FNMulAdd( dy3, dy3, FS::FNMulAdd( dx3, dx3, float32v( kFalloffRadiusSquared ) ) ) ) );
        float32v falloff4 = falloff0 + FS::FMulAdd( unskewDelta,
            float32v( -4.0 * ( kRoot5 + 3.0 ) / ( kRoot5 + 5.0 ) ),
            float32v( -4.0 / 5.0 ) );

        falloff0 = FS::Max( falloff0, float32v( 0 ) );
        falloff1 = FS::Max( falloff1, float32v( 0 ) );
        falloff2 = FS::Max( falloff2, float32v( 0 ) );
        falloff3 = FS::Max( falloff3, float32v( 0 ) );
        falloff4 = FS::Max( falloff4, float32v( 0 ) );

        falloff0 *= falloff0; falloff0 *= falloff0;
        falloff1 *= falloff1; falloff1 *= falloff1;
        falloff2 *= falloff2; falloff2 *= falloff2;
        falloff3 *= falloff3; falloff3 *= falloff3;
        falloff4 *= falloff4; falloff4 *= falloff4;

        float32v gradientRampValue0 = GetGradientDotPerlin( HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimedBase, wPrimedBase ), dx0, dy0, dz0, dw0 );
        float32v gradientRampValue1 = GetGradientDotPerlin( HashPrimes( seed,
            FS::MaskedAdd( maskX1, xPrimedBase, int32v( Primes::X ) ),
            FS::MaskedAdd( maskY1, yPrimedBase, int32v( Primes::Y ) ),
            FS::MaskedAdd( maskZ1, zPrimedBase, int32v( Primes::Z ) ),
            FS::MaskedAdd( maskW1, wPrimedBase, int32v( Primes::W ) ) ), dx1, dy1, dz1, dw1 );
        float32v gradientRampValue2 = GetGradientDotPerlin( HashPrimes( seed,
            FS::MaskedAdd( maskX2, xPrimedBase, int32v( Primes::X ) ),
            FS::MaskedAdd( maskY2, yPrimedBase, int32v( Primes::Y ) ),
            FS::MaskedAdd( maskZ2, zPrimedBase, int32v( Primes::Z ) ),
            FS::MaskedAdd( maskW2, wPrimedBase, int32v( Primes::W ) ) ), dx2, dy2, dz2, dw2 );
        float32v gradientRampValue3 = GetGradientDotPerlin( HashPrimes( seed,
            FS::MaskedAdd( maskX3, xPrimedBase, int32v( Primes::X ) ),
            FS::MaskedAdd( maskY3, yPrimedBase, int32v( Primes::Y ) ),
            FS::MaskedAdd( maskZ3, zPrimedBase, int32v( Primes::Z ) ),
            FS::MaskedAdd( maskW3, wPrimedBase, int32v( Primes::W ) ) ), dx3, dy3, dz3, dw3 );
        float32v gradientRampValue4 = GetGradientDotPerlin( HashPrimes( seed,
            xPrimedBase + int32v( Primes::X ), yPrimedBase + int32v( Primes::Y ), zPrimedBase + int32v( Primes::Z ), wPrimedBase + int32v( Primes::W ) ),
            dx4, dy4, dz4, dw4 );

        float32v value = FS::FMulAdd( gradientRampValue0, falloff0, FS::FMulAdd( gradientRampValue1, falloff1, FS::FMulAdd( gradientRampValue2, falloff2, FS::FMulAdd( gradientRampValue3, falloff3, gradientRampValue4 * falloff4 ) ) ) );

        constexpr float kBounding = 0.036730043590068817138671875f;

        return this->ScaleOutput( value, -kBounding, kBounding );
    }

public:
    static constexpr float kBounding2D = 0.0261208079755306243896484375f;
    static constexpr float kBounding3D = 0.030586399137973785400390625f;

    // Same arithmetic as Gen after ScalePositions, returns the native value and its gradient per noise cell
    static FS_FORCEINLINE float32v ValueGradient( int32v seed, float32v x, float32v y, float32v& gradX, float32v& gradY )
    {
        constexpr double kRoot3 = 1.7320508075688772935274463415059;
        constexpr double kSkew2 = 1.0 / ( kRoot3 + 1.0 );
        constexpr double kUnskew2 = -1.0 / ( kRoot3 + 3.0 );
        constexpr double kFalloffRadiusSquared = 0.5;

        float32v skewDelta = float32v( kSkew2 ) * ( x + y );
        float32v xSkewed = x + skewDelta;
        float32v ySkewed = y + skewDelta;

        float32v xSkewedBase = FS::Floor( xSkewed );
        float32v ySkewedBase = FS::Floor( ySkewed );
        float32v dxSkewed = xSkewed - xSkewedBase;
        float32v dySkewed = ySkewed - ySkewedBase;

        int32v xPrimedBase = FS::Convert<int32_t>( xSkewedBase ) * int32v( Primes::X );
        int32v yPrimedBase = FS::Convert<int32_t>( ySkewedBase ) * int32v( Primes::Y );

        mask32v xGreaterEqualY = dxSkewed >= dySkewed;

        float32v unskewDelta = float32v( kUnskew2 ) * ( dxSkewed + dySkewed );
        float32v dx0 = dxSkewed + unskewDelta;
        float32v dy0 = dySkewed + unskewDelta;

        float32v dx1 = FS::MaskedIncrement( ~xGreaterEqualY, dx0 ) - float32v( kUnskew2 + 1 );
        float32v dy1 = FS::MaskedIncrement( xGreaterEqualY, dy0 ) - float32v( kUnskew2 + 1 );
        float32v dx2 = dx0 - float32v( kUnskew2 * 2 + 1 );
        float32v dy2 = dy0 - float32v( kUnskew2 * 2 + 1 );

        float32v falloff0 = FS::FNMulAdd( dx0, dx0, FS::FNMulAdd( dy0, dy0, float32v( kFalloffRadiusSquared ) ) );
        float32v falloff1 = FS::FNMulAdd( dx1, dx1, FS::FNMulAdd( dy1, dy1, float32v( kFalloffRadiusSquared ) ) );
        float32v falloff2 = falloff0 + FS::FMulAdd( unskewDelta, float32v( -4.0 * ( kRoot3 + 2.0 ) / ( kRoot3 + 3.0 ) ), float32v( -2.0 / 3.0 ) );

        falloff0 = FS::Max( falloff0, float32v( 0 ) );
        falloff1 = FS::Max( falloff1, float32v( 0 ) );
        falloff2 = FS::Max( falloff2, float32v( 0 ) );

        float32v t0 = falloff0, t1 = falloff1, t2 = falloff2;

        falloff0 *= falloff0;
        falloff0 *= falloff0;
        falloff1 *= falloff1;
        falloff1 *= falloff1;
        falloff2 *= falloff2;
        falloff2 *= falloff2;

        int32v hash0 = HashPrimes( seed, xPrimedBase, yPrimedBase );
        int32v hash1 = HashPrimes( seed, FS::MaskedAdd( xGreaterEqualY, xPrimedBase, int32v( Primes::X ) ), FS::InvMaskedAdd( xGreaterEqualY, yPrimedBase, int32v( Primes::Y ) ) );
        int32v hash2 = HashPrimes( seed, xPrimedBase + int32v( Primes::X ), yPrimedBase + int32v( Primes::Y ) );

        float32v gradientRampValue0 = GetGradientDotPerlin( hash0, dx0, dy0 );
        float32v gradientRampValue1 = GetGradientDotPerlin( hash1, dx1, dy1 );
        float32v gradientRampValue2 = GetGradientDotPerlin( hash2, dx2, dy2 );

        float32v value = FS::FMulAdd( gradientRampValue0, falloff0, FS::FMulAdd( gradientRampValue1, falloff1, gradientRampValue2 * falloff2 ) );

        // d/dp ( t^4 * g.d ) = t^4 * g - 8 * t^3 * (g.d) * d, the unskew Jacobian is identity in 2D
        gradX = float32v( 0 );
        gradY = float32v( 0 );
        auto addVertex = [&]( int32v hash, float32v ramp, float32v t, float32v falloff, float32v dx, float32v dy ) {
            float32v radial = float32v( -8 ) * t * t * t * ramp;
            float32v gx, gy;
            GetGradientPerlin( hash, gx, gy );
            gradX += FS::FMulAdd( falloff, gx, radial * dx );
            gradY += FS::FMulAdd( falloff, gy, radial * dy );
        };
        addVertex( hash0, gradientRampValue0, t0, falloff0, dx0, dy0 );
        addVertex( hash1, gradientRampValue1, t1, falloff1, dx1, dy1 );
        addVertex( hash2, gradientRampValue2, t2, falloff2, dx2, dy2 );

        return value;
    }

    static FS_FORCEINLINE float32v ValueGradient( int32v seed, float32v x, float32v y, float32v z, float32v& gradX, float32v& gradY, float32v& gradZ )
    {
        constexpr double kSkew3 = 1.0 / 3.0;
        constexpr double kReflectUnskew3 = -1.0 / 2.0;
        constexpr double kFalloffRadiusSquared = 0.6;

        float32v skewDelta = float32v( kSkew3 ) * ( x + y + z );
        float32v xSkewed = x + skewDelta;
        float32v ySkewed = y + skewDelta;
        float32v zSkewed = z + skewDelta;

        float32v xSkewedBase = FS::Floor( xSkewed );
        float32v ySkewedBase = FS::Floor( ySkewed );
        float32v zSkewedBase = FS::Floor( zSkewed );
        float32v dxSkewed = xSkewed - xSkewedBase;
        float32v dySkewed = ySkewed - ySkewedBase;
        float32v dzSkewed = zSkewed - zSkewedBase;

        int32v xPrimedBase = FS::Convert<int32_t>( xSkewedBase ) * int32v( Primes::X );
        int32v yPrimedBase = FS::Convert<int32_t>( ySkewedBase ) * int32v( Primes::Y );
        int32v zPrimedBase = FS::Convert<int32_t>( zSkewedBase ) * int32v( Primes::Z );

        mask32v xGreaterEqualY = dxSkewed >= dySkewed;
        mask32v yGreaterEqualZ = dySkewed >= dzSkewed;
        mask32v xGreaterEqualZ = dxSkewed >= dzSkewed;

        float32v unskewDelta = float32v( kReflectUnskew3 ) * ( dxSkewed + dySkewed + dzSkewed );
        float32v dx0 = dxSkewed + unskewDelta;
        float32v dy0 = dySkewed + unskewDelta;
        float32v dz0 = dzSkewed + unskewDelta;

        mask32v maskX1 = xGreaterEqualY & xGreaterEqualZ;
        mask32v maskY1 = FS::BitwiseAndNot( yGreaterEqualZ, xGreaterEqualY );
        mask32v maskZ1 = xGreaterEqualZ | yGreaterEqualZ; // Inv masked

        mask32v nMaskX2 = xGreaterEqualY | xGreaterEqualZ; // Inv masked
        mask32v nMaskY2 = FS::BitwiseAndNot( xGreaterEqualY, yGreaterEqualZ );
        mask32v nMaskZ2 = xGreaterEqualZ & yGreaterEqualZ;

        float32v dx3 = dx0 - float32v( kReflectUnskew3 * 3 + 1 );
        float32v dy3 = dy0 - float32v( kReflectUnskew3 * 3 + 1 );
        float32v dz3 = dz0 - float32v( kReflectUnskew3 * 3 + 1 );
        float32v dx1 = FS::MaskedSub( maskX1, dx3, float32v( 1 ) );
        float32v dy1 = FS::MaskedSub( maskY1, dy3, float32v( 1 ) );
        float32v dz1 = FS::InvMaskedSub( maskZ1, dz3, float32v( 1 ) );
        float32v dx2 = FS::MaskedIncrement( ~nMaskX2, dx0 );
        float32v dy2 = FS::MaskedIncrement( nMaskY2, dy0 );
        float32v dz2 = FS::MaskedIncrement( nMaskZ2, dz0 );

        float32v falloff0 = FS::FNMulAdd( dz0, dz0, FS::FNMulAdd( dy0, dy0, FS::FNMulAdd( dx0, dx0, float32v( kFalloffRadiusSquared ) ) ) );
        float32v falloff1 = FS::FNMulAdd( dz1, dz1, FS::FNMulAdd( dy1, dy1, FS::FNMulAdd( dx1, dx1, float32v( kFalloffRadiusSquared ) ) ) );
        float32v falloff2 = FS::FNMulAdd( dz2, dz2, FS::FNMulAdd( dy2, dy2, FS::FNMulAdd( dx2, dx2, float32v( kFalloffRadiusSquared ) ) ) );
        float32v falloff3 = falloff0 - ( unskewDelta + float32v( 3.0 / 4.0 ) );

        falloff0 = FS::Max( falloff0, float32v( 0 ) );
        falloff1 = FS::Max( falloff1, float32v( 0 ) );
        falloff2 = FS::Max( falloff2, float32v( 0 ) );
        falloff3 = FS::Max( falloff3, float32v( 0 ) );

        float32v t0 = falloff0, t1 = falloff1, t2 = falloff2, t3 = falloff3;

        falloff0 *= falloff0;
        falloff0 *= falloff0;
        falloff1 *= falloff1;
        falloff1 *= falloff1;
        falloff2 *= falloff2;
        falloff2 *= falloff2;
        falloff3 *= falloff3;
        falloff3 *= falloff3;

        int32v hash0 = HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimedBase );
        int32v hash1 = HashPrimes( seed, FS::MaskedAdd( maskX1, xPrimedBase, int32v( Primes::X ) ), FS::MaskedAdd( maskY1, yPrimedBase, int32v( Primes::Y ) ), FS::InvMaskedAdd( maskZ1, zPrimedBase, int32v( Primes::Z ) ) );
        int32v hash2 = HashPrimes( seed, FS::MaskedAdd( nMaskX2, xPrimedBase, int32v( Primes::X ) ), FS::InvMaskedAdd( nMaskY2, yPrimedBase, int32v( Primes::Y ) ), FS::InvMaskedAdd( nMaskZ2, zPrimedBase, int32v( Primes::Z ) ) );
        int32v hash3 = HashPrimes( seed, xPrimedBase + int32v( Primes::X ), yPrimedBase + int32v( Primes::Y ), zPrimedBase + int32v( Primes::Z ) );

        float32v gradientRampValue0 = GetGradientDotCommon( hash0, dx0, dy0, dz0 );
        float32v gradientRampValue1 = GetGradientDotCommon( hash1, dx1, dy1, dz1 );
        float32v gradientRampValue2 = GetGradientDotCommon( hash2, dx2, dy2, dz2 );
        float32v gradientRampValue3 = GetGradientDotCommon( hash3, dx3, dy3, dz3 );

        float32v value = FS::FMulAdd( gradientRampValue3, falloff3, FS::FMulAdd( gradientRampValue2, falloff2, FS::FMulAdd( gradientRampValue1, falloff1, gradientRampValue0 * falloff0 ) ) );

        float32v gX( 0 ), gY( 0 ), gZ( 0 );
        auto addVertex = [&]( int32v hash, float32v ramp, float32v t, float32v falloff, float32v dx, float32v dy, float32v dz ) {
            float32v radial = float32v( -8 ) * t * t * t * ramp;
            float32v gx, gy, gz;
            GetGradientCommon( hash, gx, gy, gz );
            gX += FS::FMulAdd( falloff, gx, radial * dx );
            gY += FS::FMulAdd( falloff, gy, radial * dy );
            gZ += FS::FMulAdd( falloff, gz, radial * dz );
        };
        addVertex( hash0, gradientRampValue0, t0, falloff0, dx0, dy0, dz0 );
        addVertex( hash1, gradientRampValue1, t1, falloff1, dx1, dy1, dz1 );
        addVertex( hash2, gradientRampValue2, t2, falloff2, dx2, dy2, dz2 );
        addVertex( hash3, gradientRampValue3, t3, falloff3, dx3, dy3, dz3 );

        // The skew then reflected unskew is the reflection I - 2/3 * 11^T, which is its own transpose
        float32v reflect = float32v( -2.0 / 3.0 ) * ( gX + gY + gZ );
        gradX = gX + reflect;
        gradY = gY + reflect;
        gradZ = gZ + reflect;

        return value;
    }

    bool FS_VECTORCALL GenGradient( int32v seed, float32v x, float32v y, float32v& value, float32v& dx, float32v& dy ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y );

        value = ValueGradient( seed, x, y, dx, dy );

        float32v slope = this->ScaleOutputSlope( -kBounding2D, kBounding2D );
        dx *= slope;
        dy *= slope;
        value = this->ScaleOutput( value, -kBounding2D, kBounding2D );
        return true;
    }

    bool FS_VECTORCALL GenGradient( int32v seed, float32v x, float32v y, float32v z, float32v& value, float32v& dx, float32v& dy, float32v& dz ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y, z );

        value = ValueGradient( seed, x, y, z, dx, dy, dz );

        float32v slope = this->ScaleOutputSlope( -kBounding3D, kBounding3D );
        dx *= slope;
        dy *= slope;
        dz *= slope;
        value = this->ScaleOutput( value, -kBounding3D, kBounding3D );
        return true;
    }

    static constexpr float kBounding4D = 0.036730043590068817138671875f;

    static FS_FORCEINLINE float32v ValueGradient( int32v seed, float32v x, float32v y, float32v z, float32v w, float32v& gradX, float32v& gradY, float32v& gradZ, float32v& gradW )
    {
        constexpr double kRoot5 = 2.2360679774997896964091736687313;
        constexpr double kSkew4 = 1.0 / ( kRoot5 + 1.0 );
        constexpr double kUnskew4 = -1.0 / ( kRoot5 + 5.0 );
        constexpr double kFalloffRadiusSquared = 0.6;

        float32v skewDelta = float32v( kSkew4 ) * ( x + y + z + w );
        float32v xSkewed = x + skewDelta;
        float32v ySkewed = y + skewDelta;
        float32v zSkewed = z + skewDelta;
        float32v wSkewed = w + skewDelta;

        float32v xSkewedBase = FS::Floor( xSkewed );
        float32v ySkewedBase = FS::Floor( ySkewed );
        float32v zSkewedBase = FS::Floor( zSkewed );
        float32v wSkewedBase = FS::Floor( wSkewed );
        float32v dxSkewed = xSkewed - xSkewedBase;
        float32v dySkewed = ySkewed - ySkewedBase;
        float32v dzSkewed = zSkewed - zSkewedBase;
        float32v dwSkewed = wSkewed - wSkewedBase;

        int32v xPrimedBase = FS::Convert<int32_t>( xSkewedBase ) * int32v( Primes::X );
        int32v yPrimedBase = FS::Convert<int32_t>( ySkewedBase ) * int32v( Primes::Y );
        int32v zPrimedBase = FS::Convert<int32_t>( zSkewedBase ) * int32v( Primes::Z );
        int32v wPrimedBase = FS::Convert<int32_t>( wSkewedBase ) * int32v( Primes::W );

        float32v unskewDelta = float32v( kUnskew4 ) * ( dxSkewed + dySkewed + dzSkewed + dwSkewed );
        float32v dx0 = dxSkewed + unskewDelta;
        float32v dy0 = dySkewed + unskewDelta;
        float32v dz0 = dzSkewed + unskewDelta;
        float32v dw0 = dwSkewed + unskewDelta;

        int32v rankX( 0 );
        int32v rankY( 0 );
        int32v rankZ( 0 );
        int32v rankW( 0 );

        mask32v xGreaterEqualY = dx0 >= dy0;
        rankX = FS::MaskedIncrement( xGreaterEqualY, rankX );
        rankY = FS::MaskedIncrement( ~xGreaterEqualY, rankY );

        mask32v xGreaterEqualZ = dx0 >= dz0;
        rankX = FS::MaskedIncrement( xGreaterEqualZ, rankX );
        rankZ = FS::MaskedIncrement( ~xGreaterEqualZ, rankZ );

        mask32v xGreaterEqualW = dx0 >= dw0;
        rankX = FS::MaskedIncrement( xGreaterEqualW, rankX );
        rankW = FS::MaskedIncrement( ~xGreaterEqualW, rankW );

        mask32v yGreaterEqualZ = dy0 >= dz0;
        rankY = FS::MaskedIncrement( yGreaterEqualZ, rankY );
        rankZ = FS::MaskedIncrement( ~yGreaterEqualZ, rankZ );

        mask32v yGreaterEqualW = dy0 >= dw0;
        rankY = FS::MaskedIncrement( yGreaterEqualW, rankY );
        rankW = FS::MaskedIncrement( ~yGreaterEqualW, rankW );

        mask32v zGreaterEqualW = dz0 >= dw0;
        rankZ = FS::MaskedIncrement( zGreaterEqualW, rankZ );
        rankW = FS::MaskedIncrement( ~zGreaterEqualW, rankW );

        mask32v maskX1 = rankX > int32v( 2 );
        mask32v maskY1 = rankY > int32v( 2 );
        mask32v maskZ1 = rankZ > int32v( 2 );
        mask32v maskW1 = rankW > int32v( 2 );

        mask32v maskX2 = rankX > int32v( 1 );
        mask32v maskY2 = rankY > int32v( 1 );
        mask32v maskZ2 = rankZ > int32v( 1 );
        mask32v maskW2 = rankW > int32v( 1 );

        mask32v maskX3 = rankX > int32v( 0 );
        mask32v maskY3 = rankY > int32v( 0 );
        mask32v maskZ3 = rankZ > int32v( 0 );
        mask32v maskW3 = rankW > int32v( 0 );

        float32v dx1 = FS::MaskedSub( maskX1, dx0, float32v( 1 ) ) - float32v( kUnskew4 );
        float32v dy1 = FS::MaskedSub( maskY1, dy0, float32v( 1 ) ) - float32v( kUnskew4 );
        float32v dz1 = FS::MaskedSub( maskZ1, dz0, float32v( 1 ) ) - float32v( kUnskew4 );
        float32v dw1 = FS::MaskedSub( maskW1, dw0, float32v( 1 ) ) - float32v( kUnskew4 );
        float32v dx2 = FS::MaskedSub( maskX2, dx0, float32v( 1 ) ) - float32v( kUnskew4 * 2 );
        float32v dy2 = FS::MaskedSub( maskY2, dy0, float32v( 1 ) ) - float32v( kUnskew4 * 2 );
        float32v dz2 = FS::MaskedSub( maskZ2, dz0, float32v( 1 ) ) - float32v( kUnskew4 * 2 );
        float32v dw2 = FS::MaskedSub( maskW2, dw0, float32v( 1 ) ) - float32v( kUnskew4 * 2 );
        float32v dx3 = FS::MaskedSub( maskX3, dx0, float32v( 1 ) ) - float32v( kUnskew4 * 3 );
        float32v dy3 = FS::MaskedSub( maskY3, dy0, float32v( 1 ) ) - float32v( kUnskew4 * 3 );
        float32v dz3 = FS::MaskedSub( maskZ3, dz0, float32v( 1 ) ) - float32v( kUnskew4 * 3 );
        float32v dw3 = FS::MaskedSub( maskW3, dw0, float32v( 1 ) ) - float32v( kUnskew4 * 3 );
        float32v dx4 = dx0 - float32v( kUnskew4 * 4 + 1 );
        float32v dy4 = dy0 - float32v( kUnskew4 * 4 + 1 );
        float32v dz4 = dz0 - float32v( kUnskew4 * 4 + 1 );
        float32v dw4 = dw0 - float32v( kUnskew4 * 4 + 1 );

        float32v falloff0 = FS::FNMulAdd( dw0, dw0, FS::FNMulAdd( dz0, dz0, FS::FNMulAdd( dy0, dy0, FS::FNMulAdd( dx0, dx0, float32v( kFalloffRadiusSquared ) ) ) ) );
        float32v falloff1 = FS::FNMulAdd( dw1, dw1, FS::FNMulAdd( dz1, dz1, FS::FNMulAdd( dy1, dy1, FS::FNMulAdd( dx1, dx1, float32v( kFalloffRadiusSquared ) ) ) ) );
        float32v falloff2 = FS::FNMulAdd( dw2, dw2, FS::FNMulAdd( dz2, dz2, FS::FNMulAdd( dy2, dy2, FS::FNMulAdd( dx2, dx2, float32v( kFalloffRadiusSquared ) ) ) ) );
        float32v falloff3 = FS::FNMulAdd( dw3, dw3, FS::FNMulAdd( dz3, dz3, FS::FNMulAdd( dy3, dy3, FS::FNMulAdd( dx3, dx3, float32v( kFalloffRadiusSquared ) ) ) ) );
        float32v falloff4 = falloff0 + FS::FMulAdd( unskewDelta, float32v( -4.0 * ( kRoot5 + 3.0 ) / ( kRoot5 + 5.0 ) ), float32v( -4.0 / 5.0 ) );

        falloff0 = FS::Max( falloff0, float32v( 0 ) );
        falloff1 = FS::Max( falloff1, float32v( 0 ) );
        falloff2 = FS::Max( falloff2, float32v( 0 ) );
        falloff3 = FS::Max( falloff3, float32v( 0 ) );
        falloff4 = FS::Max( falloff4, float32v( 0 ) );

        float32v t0 = falloff0, t1 = falloff1, t2 = falloff2, t3 = falloff3, t4 = falloff4;

        falloff0 *= falloff0;
        falloff0 *= falloff0;
        falloff1 *= falloff1;
        falloff1 *= falloff1;
        falloff2 *= falloff2;
        falloff2 *= falloff2;
        falloff3 *= falloff3;
        falloff3 *= falloff3;
        falloff4 *= falloff4;
        falloff4 *= falloff4;

        int32v hash0 = HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimedBase, wPrimedBase );
        int32v hash1 = HashPrimes( seed, FS::MaskedAdd( maskX1, xPrimedBase, int32v( Primes::X ) ), FS::MaskedAdd( maskY1, yPrimedBase, int32v( Primes::Y ) ),
                                   FS::MaskedAdd( maskZ1, zPrimedBase, int32v( Primes::Z ) ), FS::MaskedAdd( maskW1, wPrimedBase, int32v( Primes::W ) ) );
        int32v hash2 = HashPrimes( seed, FS::MaskedAdd( maskX2, xPrimedBase, int32v( Primes::X ) ), FS::MaskedAdd( maskY2, yPrimedBase, int32v( Primes::Y ) ),
                                   FS::MaskedAdd( maskZ2, zPrimedBase, int32v( Primes::Z ) ), FS::MaskedAdd( maskW2, wPrimedBase, int32v( Primes::W ) ) );
        int32v hash3 = HashPrimes( seed, FS::MaskedAdd( maskX3, xPrimedBase, int32v( Primes::X ) ), FS::MaskedAdd( maskY3, yPrimedBase, int32v( Primes::Y ) ),
                                   FS::MaskedAdd( maskZ3, zPrimedBase, int32v( Primes::Z ) ), FS::MaskedAdd( maskW3, wPrimedBase, int32v( Primes::W ) ) );
        int32v hash4 = HashPrimes( seed, xPrimedBase + int32v( Primes::X ), yPrimedBase + int32v( Primes::Y ), zPrimedBase + int32v( Primes::Z ), wPrimedBase + int32v( Primes::W ) );

        float32v gradientRampValue0 = GetGradientDotPerlin( hash0, dx0, dy0, dz0, dw0 );
        float32v gradientRampValue1 = GetGradientDotPerlin( hash1, dx1, dy1, dz1, dw1 );
        float32v gradientRampValue2 = GetGradientDotPerlin( hash2, dx2, dy2, dz2, dw2 );
        float32v gradientRampValue3 = GetGradientDotPerlin( hash3, dx3, dy3, dz3, dw3 );
        float32v gradientRampValue4 = GetGradientDotPerlin( hash4, dx4, dy4, dz4, dw4 );

        float32v value = FS::FMulAdd( gradientRampValue0, falloff0, FS::FMulAdd( gradientRampValue1, falloff1, FS::FMulAdd( gradientRampValue2, falloff2, FS::FMulAdd( gradientRampValue3, falloff3, gradientRampValue4 * falloff4 ) ) ) );

        // Same per-vertex rule as 2D; the 4D skew and unskew also cancel to identity
        gradX = gradY = gradZ = gradW = float32v( 0 );
        auto addVertex = [&]( int32v hash, float32v ramp, float32v t, float32v falloff, float32v dx, float32v dy, float32v dz, float32v dw ) {
            float32v radial = float32v( -8 ) * t * t * t * ramp;
            float32v gx, gy, gz, gw;
            GetGradientPerlin( hash, gx, gy, gz, gw );
            gradX += FS::FMulAdd( falloff, gx, radial * dx );
            gradY += FS::FMulAdd( falloff, gy, radial * dy );
            gradZ += FS::FMulAdd( falloff, gz, radial * dz );
            gradW += FS::FMulAdd( falloff, gw, radial * dw );
        };
        addVertex( hash0, gradientRampValue0, t0, falloff0, dx0, dy0, dz0, dw0 );
        addVertex( hash1, gradientRampValue1, t1, falloff1, dx1, dy1, dz1, dw1 );
        addVertex( hash2, gradientRampValue2, t2, falloff2, dx2, dy2, dz2, dw2 );
        addVertex( hash3, gradientRampValue3, t3, falloff3, dx3, dy3, dz3, dw3 );
        addVertex( hash4, gradientRampValue4, t4, falloff4, dx4, dy4, dz4, dw4 );

        return value;
    }

    bool FS_VECTORCALL GenGradient( int32v seed, float32v x, float32v y, float32v z, float32v w, float32v& value, float32v& dx, float32v& dy, float32v& dz, float32v& dw ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y, z, w );

        value = ValueGradient( seed, x, y, z, w, dx, dy, dz, dw );

        float32v slope = this->ScaleOutputSlope( -kBounding4D, kBounding4D );
        dx *= slope;
        dy *= slope;
        dz *= slope;
        dw *= slope;
        value = this->ScaleOutput( value, -kBounding4D, kBounding4D );
        return true;
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<SuperSimplex, SIMD> final : public virtual SuperSimplex, public DispatchClass<VariableRange<Seeded<ScalableGenerator>>, SIMD>
{
    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y );

        constexpr double kRoot3 = 1.7320508075688772935274463415059;
        constexpr double kSkew2 = 1.0 / ( kRoot3 + 1.0 );
        constexpr double kUnskew2 = -1.0 / ( kRoot3 + 3.0 );
        constexpr double kFalloffRadiusSquared = 2.0 / 3.0;

        float32v skewDelta = float32v( kSkew2 ) * ( x + y );
        float32v xSkewed = x + skewDelta;
        float32v ySkewed = y + skewDelta;
        float32v xSkewedBase = FS::Floor( xSkewed );
        float32v ySkewedBase = FS::Floor( ySkewed );
        float32v dxSkewed = xSkewed - xSkewedBase;
        float32v dySkewed = ySkewed - ySkewedBase;
        int32v xPrimedBase = FS::Convert<int32_t>( xSkewedBase ) * int32v( Primes::X );
        int32v yPrimedBase = FS::Convert<int32_t>( ySkewedBase ) * int32v( Primes::Y );

        mask32v forwardXY = dxSkewed + dySkewed > float32v( 1.0f );
        float32v boundaryXY = FS::Masked( forwardXY, float32v( -1.0f ) );
        mask32v forwardX = FS::FMulAdd( dxSkewed, float32v( -2.0f ), dySkewed ) < boundaryXY;
        mask32v forwardY = FS::FMulAdd( dySkewed, float32v( -2.0f ), dxSkewed ) < boundaryXY;

        float32v unskewDelta = float32v( kUnskew2 ) * ( dxSkewed + dySkewed );
        float32v dxBase = dxSkewed + unskewDelta;
        float32v dyBase = dySkewed + unskewDelta;

        float32v falloffBase0, value;

        // Vertex <0, 0>
        {
            int32v hash = HashPrimes( seed, xPrimedBase, yPrimedBase );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dxBase, dyBase );
            falloffBase0 = FS::FNMulAdd( dxBase, dxBase, FS::FNMulAdd( dyBase, dyBase, float32v( kFalloffRadiusSquared ) ) );
            float32v falloff = falloffBase0; falloff *= falloff; falloff *= falloff;
            value = falloff * gradientRampValue;
        }

        // Vertex <1, 1>
        {
            int32v hash = HashPrimes( seed, xPrimedBase + int32v( Primes::X ), yPrimedBase + int32v( Primes::Y ) );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dxBase - float32v( 2 * kUnskew2 + 1 ), dyBase - float32v( 2 * kUnskew2 + 1 ) );
            float32v falloff = FS::FMulAdd( unskewDelta,
                float32v( -4.0 * ( kRoot3 + 2.0 ) / ( kRoot3 + 3.0 ) ),
                falloffBase0 - float32v( kFalloffRadiusSquared ) );
            falloff *= falloff; falloff *= falloff;
            value = FS::FMulAdd( falloff, gradientRampValue, value );
        }

        float32v xyDelta = FS::Select( forwardXY, float32v( kUnskew2 + 1 ), float32v( -kUnskew2 ) );
        dxBase -= xyDelta;
        dyBase -= xyDelta;

        // Vertex <1, 0> or <-1, 0> or <1, 2>
        {
            int32v hash = HashPrimes( seed,
                FS::InvMaskedSub( forwardXY, FS::MaskedAdd( forwardX, xPrimedBase, int32v( Primes::X * 2 ) ), int32v( Primes::X ) ),
                FS::MaskedAdd( forwardXY, yPrimedBase, int32v( Primes::Y ) ) );
            float32v dx = dxBase - FS::Select( forwardX, float32v( 1 + 2 * kUnskew2 ), float32v( -1 ) );
            float32v dy = FS::MaskedSub( forwardX, dyBase, float32v( 2 * kUnskew2 ) );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dx, dy );
            float32v falloff = FS::Max( FS::FNMulAdd( dx, dx, FS::FNMulAdd( dy, dy, float32v( kFalloffRadiusSquared ) ) ), float32v( 0 ) );
            falloff *= falloff; falloff *= falloff;
            value = FS::FMulAdd( falloff, gradientRampValue, value );
        }

        // Vertex <0, 1> or <0, -1> or <2, 1>
        {
            int32v hash = HashPrimes( seed,
                FS::MaskedAdd( forwardXY, xPrimedBase, int32v( Primes::X ) ),
                FS::InvMaskedSub( forwardXY, FS::MaskedAdd( forwardY, yPrimedBase, int32v( (int32_t)( Primes::Y * 2LL ) ) ), int32v( Primes::Y ) ) );
            float32v dx = FS::MaskedSub( forwardY, dxBase, float32v( 2 * kUnskew2 ) );
            float32v dy = dyBase - FS::Select( forwardY, float32v( 1 + 2 * kUnskew2 ), float32v( -1 ) );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dx, dy );
            float32v falloff = FS::Max( FS::FNMulAdd( dx, dx, FS::FNMulAdd( dy, dy, float32v( kFalloffRadiusSquared ) ) ), float32v( 0 ) );
            falloff *= falloff; falloff *= falloff;
            value = FS::FMulAdd( falloff, gradientRampValue, value );
        }

        constexpr float kBounding = 0.14084912836551666259765625f;

        return this->ScaleOutput( value, -kBounding, kBounding );
    }

    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y, z );

        constexpr double kSkew3 = 1.0 / 3.0;
        constexpr double kReflectUnskew3 = -1.0 / 2.0;
        constexpr double kTwiceUnskew3 = -1.0 / 4.0;

        constexpr double kDistanceSquaredA = 3.0 / 4.0;
        constexpr double kDistanceSquaredB = 1.0;
        constexpr double kFalloffRadiusSquared = kDistanceSquaredA;

        float32v skewDelta = float32v( kSkew3 ) * ( x + y + z );

        float32v xSkewed = x + skewDelta;
        float32v ySkewed = y + skewDelta;
        float32v zSkewed = z + skewDelta;
        float32v xSkewedBase = FS::Floor( xSkewed );
        float32v ySkewedBase = FS::Floor( ySkewed );
        float32v zSkewedBase = FS::Floor( zSkewed );
        float32v dxSkewed = xSkewed - xSkewedBase;
        float32v dySkewed = ySkewed - ySkewedBase;
        float32v dzSkewed = zSkewed - zSkewedBase;

        // From unit cell base, find closest vertex
        {
            // Perform a double unskew to get the vector whose dot product with skewed vectors produces the unskewed result.
            float32v twiceUnskewDelta = float32v( kTwiceUnskew3 ) * ( dxSkewed + dySkewed + dzSkewed );
            float32v xNormal = dxSkewed + twiceUnskewDelta;
            float32v yNormal = dySkewed + twiceUnskewDelta;
            float32v zNormal = dzSkewed + twiceUnskewDelta;
            float32v xyzNormal = -twiceUnskewDelta; // xNormal + yNormal + zNormal

            // Using those, compare scores to determine which vertex is closest.
            constexpr auto considerVertex = [] ( float32v& maxScore, int32v& moveMaskBits, float32v score, int32v bits ) constexpr
            {
                moveMaskBits = FS::Select( score > maxScore, bits, moveMaskBits );
                maxScore = FS::Max( maxScore, score );
            };
            float32v maxScore = float32v( 0.375f );
            int32v moveMaskBits = FS::Masked( xyzNormal > maxScore, int32v( -1 ) );
            maxScore = FS::Max( maxScore, xyzNormal );
            considerVertex( maxScore, moveMaskBits, xNormal, 0b001 );
            considerVertex( maxScore, moveMaskBits, yNormal, 0b010 );
            considerVertex( maxScore, moveMaskBits, zNormal, 0b100 );
            maxScore += float32v( 0.125f ) - xyzNormal;
            considerVertex( maxScore, moveMaskBits, -zNormal, 0b011 );
            considerVertex( maxScore, moveMaskBits, -yNormal, 0b101 );
            considerVertex( maxScore, moveMaskBits, -xNormal, 0b110 );

            mask32v moveX = ( moveMaskBits & int32v( 0b001 ) ) != int32v( 0 );
            mask32v moveY = ( moveMaskBits & int32v( 0b010 ) ) != int32v( 0 );
            mask32v moveZ = ( moveMaskBits & int32v( 0b100 ) ) != int32v( 0 );

            xSkewedBase = FS::MaskedIncrement( moveX, xSkewedBase );
            ySkewedBase = FS::MaskedIncrement( moveY, ySkewedBase );
            zSkewedBase = FS::MaskedIncrement( moveZ, zSkewedBase );

            dxSkewed = FS::MaskedDecrement( moveX, dxSkewed );
            dySkewed = FS::MaskedDecrement( moveY, dySkewed );
            dzSkewed = FS::MaskedDecrement( moveZ, dzSkewed );
        }

        int32v xPrimedBase = FS::Convert<int32_t>( xSkewedBase ) * int32v( Primes::X );
        int32v yPrimedBase = FS::Convert<int32_t>( ySkewedBase ) * int32v( Primes::Y );
        int32v zPrimedBase = FS::Convert<int32_t>( zSkewedBase ) * int32v( Primes::Z );

        float32v skewedCoordinateSum = dxSkewed + dySkewed + dzSkewed;
        float32v twiceUnskewDelta = float32v( kTwiceUnskew3 ) * skewedCoordinateSum;
        float32v xNormal = dxSkewed + twiceUnskewDelta;
        float32v yNormal = dySkewed + twiceUnskewDelta;
        float32v zNormal = dzSkewed + twiceUnskewDelta;
        float32v xyzNormal = -twiceUnskewDelta; // xNormal + yNormal + zNormal

        float32v unskewDelta = float32v( kReflectUnskew3 ) * skewedCoordinateSum;
        float32v dxBase = dxSkewed + unskewDelta;
        float32v dyBase = dySkewed + unskewDelta;
        float32v dzBase = dzSkewed + unskewDelta;

        float32v coordinateSum = float32v( 1 + 3 * kReflectUnskew3 ) * skewedCoordinateSum; // dxBase + dyBase + dzBase

        // Vertex <0, 0, 0>
        float32v value, falloffBaseStemA, falloffBaseStemB;
        {
            float32v gradientRampValue = GetGradientDotCommon( HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimedBase ), dxBase, dyBase, dzBase );
            float32v falloffBase = FS::FNMulAdd( dzBase, dzBase, FS::FNMulAdd( dyBase, dyBase, FS::FNMulAdd( dxBase, dxBase, float32v( kFalloffRadiusSquared ) ) ) ) * float32v( 0.5f );
            falloffBaseStemA = falloffBase - float32v( kDistanceSquaredA * 0.5 );
            falloffBaseStemB = falloffBase - float32v( kDistanceSquaredB * 0.5 );
            value = ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ) * gradientRampValue;
        }

        // Vertex <1, 1, 1> or <-1, -1, -1>
        {
            mask32v signMask = xyzNormal < float32v( 0 );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset = float32v( 3 * kReflectUnskew3 + 1 ) ^ sign;

            float32v gradientRampValue = GetGradientDotCommon( HashPrimes( seed, xPrimed, yPrimed, zPrimed ), dxBase - offset, dyBase - offset, dzBase - offset );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset, coordinateSum, falloffBaseStemA ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <1, 1, 0> or <-1, -1, 0>
        {
            mask32v signMask = xyzNormal < zNormal;

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset0 = float32v( 2 * kReflectUnskew3 ) ^ sign;

            float32v gradientRampValue = GetGradientDotCommon( HashPrimes( seed, xPrimed, yPrimed, zPrimedBase ), dxBase, dyBase, dzBase - offset0 );
            float32v falloffBase = FS::Min( ( sign ^ dzBase ) - falloffBaseStemB, float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <1, 0, 1> or <-1, 0, -1>
        {
            mask32v signMask = xyzNormal < yNormal;

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset0 = float32v( 2 * kReflectUnskew3 ) ^ sign;

            float32v gradientRampValue = GetGradientDotCommon( HashPrimes( seed, xPrimed, yPrimedBase, zPrimed ), dxBase, dyBase - offset0, dzBase );
            float32v falloffBase = FS::Min( ( sign ^ dyBase ) - falloffBaseStemB, float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <0, 1, 1> or <0, -1, -1>
        {
            mask32v signMask = xyzNormal < xNormal;

            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset0 = float32v( 2 * kReflectUnskew3 ) ^ sign;

            float32v gradientRampValue = GetGradientDotCommon( HashPrimes( seed, xPrimedBase, yPrimed, zPrimed ), dxBase - offset0, dyBase, dzBase );
            float32v falloffBase = FS::Min( ( sign ^ dxBase ) - falloffBaseStemB, float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <1, 0, 0> or <-1, 0, 0>
        {
            mask32v signMask = xNormal < float32v( 0 );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset0 = float32v( kReflectUnskew3 ) ^ sign; // offset1 = -offset0 because kReflectUnskew3 + 1 = -kReflectUnskew3

            float32v gradientRampValue = GetGradientDotCommon( HashPrimes( seed, xPrimed, yPrimedBase, zPrimedBase ), dxBase + offset0, dyBase - offset0, dzBase - offset0 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemA ) + ( sign ^ dxBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <0, 1, 0> or <0, -1, 0>
        {
            mask32v signMask = yNormal < float32v( 0 );

            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset0 = float32v( kReflectUnskew3 ) ^ sign; // offset1 = -offset0 because kReflectUnskew3 + 1 = -kReflectUnskew3

            float32v gradientRampValue = GetGradientDotCommon( HashPrimes( seed, xPrimedBase, yPrimed, zPrimedBase ), dxBase - offset0, dyBase + offset0, dzBase - offset0 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemA ) + ( sign ^ dyBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <0, 0, 1> or <0, 0, -1>
        {
            mask32v signMask = zNormal < float32v( 0 );

            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset0 = float32v( kReflectUnskew3 ) ^ sign; // offset1 = -offset0 because kReflectUnskew3 + 1 = -kReflectUnskew3

            float32v gradientRampValue = GetGradientDotCommon( HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimed ), dxBase - offset0, dyBase - offset0, dzBase + offset0 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemA ) + ( sign ^ dzBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        constexpr float kBounding = 0.0069091119803488254547119140625f;
        return this->ScaleOutput( value, -kBounding, kBounding );
    }

    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z, float32v w ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y, z, w );

        constexpr double kRoot5 = 2.2360679774997896964091736687313;
        constexpr double kSkew4 = 1.0 / ( kRoot5 + 1.0 );
        constexpr double kUnskew4 = -1.0 / ( kRoot5 + 5.0 );
        constexpr double kTwiceUnskew4 = -1.0 / 5.0;

        constexpr double kDistanceSquaredA = 4.0 / 5.0;
        constexpr double kDistanceSquaredB = 6.0 / 5.0;
        constexpr double kFalloffRadiusSquared = kDistanceSquaredA;

        float32v skewDelta = float32v( kSkew4 ) * ( x + y + z + w );

        float32v xSkewed = x + skewDelta;
        float32v ySkewed = y + skewDelta;
        float32v zSkewed = z + skewDelta;
        float32v wSkewed = w + skewDelta;
        float32v xSkewedBase = FS::Floor( xSkewed );
        float32v ySkewedBase = FS::Floor( ySkewed );
        float32v zSkewedBase = FS::Floor( zSkewed );
        float32v wSkewedBase = FS::Floor( wSkewed );
        float32v dxSkewed = xSkewed - xSkewedBase;
        float32v dySkewed = ySkewed - ySkewedBase;
        float32v dzSkewed = zSkewed - zSkewedBase;
        float32v dwSkewed = wSkewed - wSkewedBase;

        // From unit cell base, find closest vertex
        {
            // Perform a double unskew to get the vector whose dot product with skewed vectors produces the unskewed result.
            float32v twiceUnskewDelta = float32v( kTwiceUnskew4 ) * ( dxSkewed + dySkewed + dzSkewed + dwSkewed );
            float32v xNormal = dxSkewed + twiceUnskewDelta;
            float32v yNormal = dySkewed + twiceUnskewDelta;
            float32v zNormal = dzSkewed + twiceUnskewDelta;
            float32v wNormal = dwSkewed + twiceUnskewDelta;
            float32v xyzwNormal = -twiceUnskewDelta; // xNormal + yNormal + zNormal + wNormal

            // Using those, compare scores to determine which vertex is closest.
            constexpr auto considerVertex = [] ( float32v& maxScore, int32v& moveMaskBits, float32v score, int32v bits ) constexpr
            {
                moveMaskBits = FS::Select( score > maxScore, bits, moveMaskBits );
                maxScore = FS::Max( maxScore, score );
            };
            float32v maxScore = float32v( 0.6f ) - xyzwNormal;
            int32v moveMaskBits = FS::Masked( float32v( 0.2f ) > maxScore, int32v( -1 ) );
            maxScore = FS::Max( maxScore, float32v( 0.2f ) );
            considerVertex( maxScore, moveMaskBits, -wNormal, 0b0111 );
            considerVertex( maxScore, moveMaskBits, -zNormal, 0b1011 );
            considerVertex( maxScore, moveMaskBits, -yNormal, 0b1101 );
            considerVertex( maxScore, moveMaskBits, -xNormal, 0b1110 );
            maxScore += xyzwNormal - float32v( 0.2f );
            considerVertex( maxScore, moveMaskBits, xNormal, 0b0001 );
            considerVertex( maxScore, moveMaskBits, yNormal, 0b0010 );
            considerVertex( maxScore, moveMaskBits, zNormal, 0b0100 );
            considerVertex( maxScore, moveMaskBits, wNormal, 0b1000 );
            maxScore += float32v( 0.2f ) - xNormal;
            considerVertex( maxScore, moveMaskBits, yNormal, 0b0011 );
            considerVertex( maxScore, moveMaskBits, zNormal, 0b0101 );
            considerVertex( maxScore, moveMaskBits, wNormal, 0b1001 );
            maxScore += xNormal;
            considerVertex( maxScore, moveMaskBits, yNormal + zNormal, 0b0110 );
            maxScore -= wNormal;
            considerVertex( maxScore, moveMaskBits, yNormal, 0b1010 );
            considerVertex( maxScore, moveMaskBits, zNormal, 0b1100 );

            mask32v moveX = ( moveMaskBits & int32v( 0b0001 ) ) != int32v( 0 );
            mask32v moveY = ( moveMaskBits & int32v( 0b0010 ) ) != int32v( 0 );
            mask32v moveZ = ( moveMaskBits & int32v( 0b0100 ) ) != int32v( 0 );
            mask32v moveW = ( moveMaskBits & int32v( 0b1000 ) ) != int32v( 0 );

            xSkewedBase = FS::MaskedIncrement( moveX, xSkewedBase );
            ySkewedBase = FS::MaskedIncrement( moveY, ySkewedBase );
            zSkewedBase = FS::MaskedIncrement( moveZ, zSkewedBase );
            wSkewedBase = FS::MaskedIncrement( moveW, wSkewedBase );

            dxSkewed = FS::MaskedDecrement( moveX, dxSkewed );
            dySkewed = FS::MaskedDecrement( moveY, dySkewed );
            dzSkewed = FS::MaskedDecrement( moveZ, dzSkewed );
            dwSkewed = FS::MaskedDecrement( moveW, dwSkewed );
        }

        int32v xPrimedBase = FS::Convert<int32_t>( xSkewedBase ) * int32v( Primes::X );
        int32v yPrimedBase = FS::Convert<int32_t>( ySkewedBase ) * int32v( Primes::Y );
        int32v zPrimedBase = FS::Convert<int32_t>( zSkewedBase ) * int32v( Primes::Z );
        int32v wPrimedBase = FS::Convert<int32_t>( wSkewedBase ) * int32v( Primes::W );

        float32v skewedCoordinateSum = dxSkewed + dySkewed + dzSkewed + dwSkewed;
        float32v twiceUnskewDelta = float32v( kTwiceUnskew4 ) * skewedCoordinateSum;
        float32v xNormal = dxSkewed + twiceUnskewDelta;
        float32v yNormal = dySkewed + twiceUnskewDelta;
        float32v zNormal = dzSkewed + twiceUnskewDelta;
        float32v wNormal = dwSkewed + twiceUnskewDelta;
        float32v xyzwNormal = -twiceUnskewDelta; // xNormal + yNormal + zNormal + wNormal

        float32v unskewDelta = float32v( kUnskew4 ) * skewedCoordinateSum;
        float32v dxBase = dxSkewed + unskewDelta;
        float32v dyBase = dySkewed + unskewDelta;
        float32v dzBase = dzSkewed + unskewDelta;
        float32v dwBase = dwSkewed + unskewDelta;

        float32v coordinateSum = float32v( 1 + 4 * kUnskew4 ) * skewedCoordinateSum; // dxBase + dyBase + dzBase + dwBase

        // Vertex <0, 0, 0, 0>
        float32v value, falloffBaseStemA, falloffBaseStemB;
        {
            float32v gradientRampValue = GetGradientDotPerlin( HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimedBase, wPrimedBase ), dxBase, dyBase, dzBase, dwBase );
            float32v falloffBase = FS::FNMulAdd( dwBase, dwBase, FS::FNMulAdd( dzBase, dzBase, FS::FNMulAdd( dyBase, dyBase, FS::FNMulAdd( dxBase, dxBase, float32v( kFalloffRadiusSquared ) ) ) ) ) * float32v( 0.5f );
            falloffBaseStemA = falloffBase - float32v( kDistanceSquaredA * 0.5 );
            falloffBaseStemB = falloffBase - float32v( kDistanceSquaredB * 0.5 );
            value = ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ) * gradientRampValue;
        }

        // Vertex <1, 1, 1, 1> or <-1, -1, -1, -1>
        {
            mask32v signMask = xyzwNormal < float32v( 0 );
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset = float32v( 4 * kUnskew4 + 1 ) ^ sign;

            float32v gradientRampValue = GetGradientDotPerlin( HashPrimes( seed, xPrimed, yPrimed, zPrimed, wPrimed ), dxBase - offset, dyBase - offset, dzBase - offset, dwBase - offset );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset, coordinateSum, falloffBaseStemA ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <1, 1, 1, 0> or <-1, -1, -1, 0>
        {
            mask32v signMask = xyzwNormal < wNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v offset1 = float32v( 3 * kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( 3 * kUnskew4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotPerlin( HashPrimes( seed, xPrimed, yPrimed, zPrimed, wPrimedBase ), dxBase - offset1, dyBase - offset1, dzBase - offset1, dwBase - offset0 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset1, coordinateSum, falloffBaseStemB ) - ( sign ^ dwBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <1, 1, 0, 1> or <-1, -1, 0, -1>
        {
            mask32v signMask = xyzwNormal < zNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( 3 * kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( 3 * kUnskew4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotPerlin( HashPrimes( seed, xPrimed, yPrimed, zPrimedBase, wPrimed ), dxBase - offset1, dyBase - offset1, dzBase - offset0, dwBase - offset1 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset1, coordinateSum, falloffBaseStemB ) - ( sign ^ dzBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <1, 0, 1, 1> or <-1, 0, -1, -1>
        {
            mask32v signMask = xyzwNormal < yNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( 3 * kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( 3 * kUnskew4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotPerlin( HashPrimes( seed, xPrimed, yPrimedBase, zPrimed, wPrimed ), dxBase - offset1, dyBase - offset0, dzBase - offset1, dwBase - offset1 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset1, coordinateSum, falloffBaseStemB ) - ( sign ^ dyBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <0, 1, 1, 1> or <0, -1, -1, -1>
        {
            mask32v signMask = xyzwNormal < xNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( 3 * kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( 3 * kUnskew4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotPerlin( HashPrimes( seed, xPrimedBase, yPrimed, zPrimed, wPrimed ), dxBase - offset0, dyBase - offset1, dzBase - offset1, dwBase - offset1 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset1, coordinateSum, falloffBaseStemB ) - ( sign ^ dxBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <1, 0, 0, 0> or <-1, 0, 0, 0>
        {
            mask32v signMask = xNormal < float32v( 0 );
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );

            float32v offset1 = float32v( kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( kUnskew4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotPerlin( HashPrimes( seed, xPrimed, yPrimedBase, zPrimedBase, wPrimedBase ), dxBase - offset1, dyBase - offset0, dzBase - offset0, dwBase - offset0 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemA ) + ( sign ^ dxBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <1, 1, 0, 0> or <-1, -1, 0, 0>
        {
            mask32v signMask = xNormal < -yNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );

            float32v offset1 = float32v( 2 * kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( 2 * kUnskew4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotPerlin( HashPrimes( seed, xPrimed, yPrimed, zPrimedBase, wPrimedBase ), dxBase - offset1, dyBase - offset1, dzBase - offset0, dwBase - offset0 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemB ) + ( sign ^ ( dxBase + dyBase ) ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <1, 0, 1, 0> or <-1, 0, -1, 0>
        {
            mask32v signMask = xNormal < -zNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v offset1 = float32v( 2 * kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( 2 * kUnskew4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotPerlin( HashPrimes( seed, xPrimed, yPrimedBase, zPrimed, wPrimedBase ), dxBase - offset1, dyBase - offset0, dzBase - offset1, dwBase - offset0 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemB ) + ( sign ^ ( dxBase + dzBase ) ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <1, 0, 0, 1> or <-1, 0, 0, -1>
        {
            mask32v signMask = xNormal < -wNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( 2 * kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( 2 * kUnskew4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotPerlin( HashPrimes( seed, xPrimed, yPrimedBase, zPrimedBase, wPrimed ), dxBase - offset1, dyBase - offset0, dzBase - offset0, dwBase - offset1 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemB ) + ( sign ^ ( dxBase + dwBase ) ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <0, 1, 0, 0> or <0, -1, 0, 0>
        {
            mask32v signMask = yNormal < float32v( 0 );
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );

            float32v offset1 = float32v( kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( kUnskew4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotPerlin( HashPrimes( seed, xPrimedBase, yPrimed, zPrimedBase, wPrimedBase ), dxBase - offset0, dyBase - offset1, dzBase - offset0, dwBase - offset0 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemA ) + ( sign ^ dyBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <0, 1, 1, 0> or <0, -1, -1, 0>
        {
            mask32v signMask = yNormal < -zNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v offset1 = float32v( 2 * kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( 2 * kUnskew4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotPerlin( HashPrimes( seed, xPrimedBase, yPrimed, zPrimed, wPrimedBase ), dxBase - offset0, dyBase - offset1, dzBase - offset1, dwBase - offset0 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemB ) + ( sign ^ ( dyBase + dzBase ) ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <0, 1, 0, 1> or <0, -1, 0, -1>
        {
            mask32v signMask = yNormal < -wNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( 2 * kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( 2 * kUnskew4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotPerlin( HashPrimes( seed, xPrimedBase, yPrimed, zPrimedBase, wPrimed ), dxBase - offset0, dyBase - offset1, dzBase - offset0, dwBase - offset1 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemB ) + ( sign ^ ( dyBase + dwBase ) ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <0, 0, 1, 0> or <0, 0, -1, 0>
        {
            mask32v signMask = zNormal < float32v( 0 );
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v offset1 = float32v( kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( kUnskew4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotPerlin( HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimed, wPrimedBase ), dxBase - offset0, dyBase - offset0, dzBase - offset1, dwBase - offset0 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemA ) + ( sign ^ dzBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <0, 0, 1, 1> or <0, 0, -1, -1>
        {
            mask32v signMask = zNormal < -wNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( 2 * kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( 2 * kUnskew4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotPerlin( HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimed, wPrimed ), dxBase - offset0, dyBase - offset0, dzBase - offset1, dwBase - offset1 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemB ) + ( sign ^ ( dzBase + dwBase ) ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        // Vertex <0, 0, 0, 1> or <0, 0, 0, -1>
        {
            mask32v signMask = wNormal < float32v( 0 );
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( kUnskew4 ) ^ sign;

            float32v gradientRampValue = GetGradientDotPerlin( HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimedBase, wPrimed ), dxBase - offset0, dyBase - offset0, dzBase - offset0, dwBase - offset1 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemA ) + ( sign ^ dwBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
        }

        constexpr float kBounding = 0.011310013942420482635498046875;

        return this->ScaleOutput( value, -kBounding, kBounding );
    }

public:
    static constexpr float kBounding2D = 0.14084912836551666259765625f;
    static constexpr float kBounding3D = 0.0069091119803488254547119140625f;

    // Same arithmetic as Gen after ScalePositions, returns the native value and its gradient per noise cell
    static FS_FORCEINLINE float32v ValueGradient( int32v seed, float32v x, float32v y, float32v& gradX, float32v& gradY )
    {
        constexpr double kRoot3 = 1.7320508075688772935274463415059;
        constexpr double kSkew2 = 1.0 / ( kRoot3 + 1.0 );
        constexpr double kUnskew2 = -1.0 / ( kRoot3 + 3.0 );
        constexpr double kFalloffRadiusSquared = 2.0 / 3.0;

        float32v skewDelta = float32v( kSkew2 ) * ( x + y );
        float32v xSkewed = x + skewDelta;
        float32v ySkewed = y + skewDelta;
        float32v xSkewedBase = FS::Floor( xSkewed );
        float32v ySkewedBase = FS::Floor( ySkewed );
        float32v dxSkewed = xSkewed - xSkewedBase;
        float32v dySkewed = ySkewed - ySkewedBase;
        int32v xPrimedBase = FS::Convert<int32_t>( xSkewedBase ) * int32v( Primes::X );
        int32v yPrimedBase = FS::Convert<int32_t>( ySkewedBase ) * int32v( Primes::Y );

        mask32v forwardXY = dxSkewed + dySkewed > float32v( 1.0f );
        float32v boundaryXY = FS::Masked( forwardXY, float32v( -1.0f ) );
        mask32v forwardX = FS::FMulAdd( dxSkewed, float32v( -2.0f ), dySkewed ) < boundaryXY;
        mask32v forwardY = FS::FMulAdd( dySkewed, float32v( -2.0f ), dxSkewed ) < boundaryXY;

        float32v unskewDelta = float32v( kUnskew2 ) * ( dxSkewed + dySkewed );
        float32v dxBase = dxSkewed + unskewDelta;
        float32v dyBase = dySkewed + unskewDelta;

        float32v falloffBase0, value;

        // t^4 * g.d contributes t^4 * g - 8 * t^3 * (g.d) * d, where t = r^2 - |d|^2; the unskew Jacobian is identity in 2D
        gradX = float32v( 0 );
        gradY = float32v( 0 );
        auto addVertex = [&]( int32v hash, float32v ramp, float32v t, float32v dx, float32v dy ) {
            float32v t2 = t * t;
            float32v radial = float32v( -8 ) * t2 * t * ramp;
            float32v gx, gy;
            GetGradientPerlin( hash, gx, gy );
            gradX += FS::FMulAdd( t2 * t2, gx, radial * dx );
            gradY += FS::FMulAdd( t2 * t2, gy, radial * dy );
        };

        // Vertex <0, 0>
        {
            int32v hash = HashPrimes( seed, xPrimedBase, yPrimedBase );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dxBase, dyBase );
            falloffBase0 = FS::FNMulAdd( dxBase, dxBase, FS::FNMulAdd( dyBase, dyBase, float32v( kFalloffRadiusSquared ) ) );
            float32v falloff = falloffBase0;
            falloff *= falloff;
            falloff *= falloff;
            value = falloff * gradientRampValue;
            addVertex( hash, gradientRampValue, falloffBase0, dxBase, dyBase );
        }

        // Vertex <1, 1>
        {
            int32v hash = HashPrimes( seed, xPrimedBase + int32v( Primes::X ), yPrimedBase + int32v( Primes::Y ) );
            float32v dx = dxBase - float32v( 2 * kUnskew2 + 1 );
            float32v dy = dyBase - float32v( 2 * kUnskew2 + 1 );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dx, dy );
            float32v t = FS::FMulAdd( unskewDelta,
                                      float32v( -4.0 * ( kRoot3 + 2.0 ) / ( kRoot3 + 3.0 ) ),
                                      falloffBase0 - float32v( kFalloffRadiusSquared ) );
            float32v falloff = t;
            falloff *= falloff;
            falloff *= falloff;
            value = FS::FMulAdd( falloff, gradientRampValue, value );
            addVertex( hash, gradientRampValue, t, dx, dy );
        }

        float32v xyDelta = FS::Select( forwardXY, float32v( kUnskew2 + 1 ), float32v( -kUnskew2 ) );
        dxBase -= xyDelta;
        dyBase -= xyDelta;

        // Vertex <1, 0> or <-1, 0> or <1, 2>
        {
            int32v hash = HashPrimes( seed,
                                      FS::InvMaskedSub( forwardXY, FS::MaskedAdd( forwardX, xPrimedBase, int32v( Primes::X * 2 ) ), int32v( Primes::X ) ),
                                      FS::MaskedAdd( forwardXY, yPrimedBase, int32v( Primes::Y ) ) );
            float32v dx = dxBase - FS::Select( forwardX, float32v( 1 + 2 * kUnskew2 ), float32v( -1 ) );
            float32v dy = FS::MaskedSub( forwardX, dyBase, float32v( 2 * kUnskew2 ) );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dx, dy );
            float32v t = FS::Max( FS::FNMulAdd( dx, dx, FS::FNMulAdd( dy, dy, float32v( kFalloffRadiusSquared ) ) ), float32v( 0 ) );
            float32v falloff = t;
            falloff *= falloff;
            falloff *= falloff;
            value = FS::FMulAdd( falloff, gradientRampValue, value );
            addVertex( hash, gradientRampValue, t, dx, dy );
        }

        // Vertex <0, 1> or <0, -1> or <2, 1>
        {
            int32v hash = HashPrimes( seed,
                                      FS::MaskedAdd( forwardXY, xPrimedBase, int32v( Primes::X ) ),
                                      FS::InvMaskedSub( forwardXY, FS::MaskedAdd( forwardY, yPrimedBase, int32v( (int32_t)( Primes::Y * 2LL ) ) ), int32v( Primes::Y ) ) );
            float32v dx = FS::MaskedSub( forwardY, dxBase, float32v( 2 * kUnskew2 ) );
            float32v dy = dyBase - FS::Select( forwardY, float32v( 1 + 2 * kUnskew2 ), float32v( -1 ) );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dx, dy );
            float32v t = FS::Max( FS::FNMulAdd( dx, dx, FS::FNMulAdd( dy, dy, float32v( kFalloffRadiusSquared ) ) ), float32v( 0 ) );
            float32v falloff = t;
            falloff *= falloff;
            falloff *= falloff;
            value = FS::FMulAdd( falloff, gradientRampValue, value );
            addVertex( hash, gradientRampValue, t, dx, dy );
        }

        return value;
    }

    static FS_FORCEINLINE float32v ValueGradient( int32v seed, float32v x, float32v y, float32v z, float32v& gradX, float32v& gradY, float32v& gradZ )
    {
        constexpr double kSkew3 = 1.0 / 3.0;
        constexpr double kReflectUnskew3 = -1.0 / 2.0;
        constexpr double kTwiceUnskew3 = -1.0 / 4.0;

        constexpr double kDistanceSquaredA = 3.0 / 4.0;
        constexpr double kDistanceSquaredB = 1.0;
        constexpr double kFalloffRadiusSquared = kDistanceSquaredA;

        float32v skewDelta = float32v( kSkew3 ) * ( x + y + z );

        float32v xSkewed = x + skewDelta;
        float32v ySkewed = y + skewDelta;
        float32v zSkewed = z + skewDelta;
        float32v xSkewedBase = FS::Floor( xSkewed );
        float32v ySkewedBase = FS::Floor( ySkewed );
        float32v zSkewedBase = FS::Floor( zSkewed );
        float32v dxSkewed = xSkewed - xSkewedBase;
        float32v dySkewed = ySkewed - ySkewedBase;
        float32v dzSkewed = zSkewed - zSkewedBase;

        // From unit cell base, find closest vertex
        {
            float32v twiceUnskewDelta = float32v( kTwiceUnskew3 ) * ( dxSkewed + dySkewed + dzSkewed );
            float32v xNormal = dxSkewed + twiceUnskewDelta;
            float32v yNormal = dySkewed + twiceUnskewDelta;
            float32v zNormal = dzSkewed + twiceUnskewDelta;
            float32v xyzNormal = -twiceUnskewDelta;

            constexpr auto considerVertex = []( float32v& maxScore, int32v& moveMaskBits, float32v score, int32v bits ) constexpr {
                moveMaskBits = FS::Select( score > maxScore, bits, moveMaskBits );
                maxScore = FS::Max( maxScore, score );
            };
            float32v maxScore = float32v( 0.375f );
            int32v moveMaskBits = FS::Masked( xyzNormal > maxScore, int32v( -1 ) );
            maxScore = FS::Max( maxScore, xyzNormal );
            considerVertex( maxScore, moveMaskBits, xNormal, 0b001 );
            considerVertex( maxScore, moveMaskBits, yNormal, 0b010 );
            considerVertex( maxScore, moveMaskBits, zNormal, 0b100 );
            maxScore += float32v( 0.125f ) - xyzNormal;
            considerVertex( maxScore, moveMaskBits, -zNormal, 0b011 );
            considerVertex( maxScore, moveMaskBits, -yNormal, 0b101 );
            considerVertex( maxScore, moveMaskBits, -xNormal, 0b110 );

            mask32v moveX = ( moveMaskBits & int32v( 0b001 ) ) != int32v( 0 );
            mask32v moveY = ( moveMaskBits & int32v( 0b010 ) ) != int32v( 0 );
            mask32v moveZ = ( moveMaskBits & int32v( 0b100 ) ) != int32v( 0 );

            xSkewedBase = FS::MaskedIncrement( moveX, xSkewedBase );
            ySkewedBase = FS::MaskedIncrement( moveY, ySkewedBase );
            zSkewedBase = FS::MaskedIncrement( moveZ, zSkewedBase );

            dxSkewed = FS::MaskedDecrement( moveX, dxSkewed );
            dySkewed = FS::MaskedDecrement( moveY, dySkewed );
            dzSkewed = FS::MaskedDecrement( moveZ, dzSkewed );
        }

        int32v xPrimedBase = FS::Convert<int32_t>( xSkewedBase ) * int32v( Primes::X );
        int32v yPrimedBase = FS::Convert<int32_t>( ySkewedBase ) * int32v( Primes::Y );
        int32v zPrimedBase = FS::Convert<int32_t>( zSkewedBase ) * int32v( Primes::Z );

        float32v skewedCoordinateSum = dxSkewed + dySkewed + dzSkewed;
        float32v twiceUnskewDelta = float32v( kTwiceUnskew3 ) * skewedCoordinateSum;
        float32v xNormal = dxSkewed + twiceUnskewDelta;
        float32v yNormal = dySkewed + twiceUnskewDelta;
        float32v zNormal = dzSkewed + twiceUnskewDelta;
        float32v xyzNormal = -twiceUnskewDelta;

        float32v unskewDelta = float32v( kReflectUnskew3 ) * skewedCoordinateSum;
        float32v dxBase = dxSkewed + unskewDelta;
        float32v dyBase = dySkewed + unskewDelta;
        float32v dzBase = dzSkewed + unskewDelta;

        float32v coordinateSum = float32v( 1 + 3 * kReflectUnskew3 ) * skewedCoordinateSum;

        // Each falloff is +-(r^2 - |d|^2) / 2, so f^4 * g.d contributes f^4 * g - 4 * |f|^3 * (g.d) * d
        float32v gX( 0 ), gY( 0 ), gZ( 0 );
        auto addVertex = [&]( int32v hash, float32v ramp, float32v falloffBase, float32v dx, float32v dy, float32v dz ) {
            float32v f = FS::Abs( falloffBase );
            float32v f2 = f * f;
            float32v radial = float32v( -4 ) * f2 * f * ramp;
            float32v gx, gy, gz;
            GetGradientCommon( hash, gx, gy, gz );
            gX += FS::FMulAdd( f2 * f2, gx, radial * dx );
            gY += FS::FMulAdd( f2 * f2, gy, radial * dy );
            gZ += FS::FMulAdd( f2 * f2, gz, radial * dz );
        };

        // Vertex <0, 0, 0>
        float32v value, falloffBaseStemA, falloffBaseStemB;
        {
            int32v hash = HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimedBase );
            float32v gradientRampValue = GetGradientDotCommon( hash, dxBase, dyBase, dzBase );
            float32v falloffBase = FS::FNMulAdd( dzBase, dzBase, FS::FNMulAdd( dyBase, dyBase, FS::FNMulAdd( dxBase, dxBase, float32v( kFalloffRadiusSquared ) ) ) ) * float32v( 0.5f );
            falloffBaseStemA = falloffBase - float32v( kDistanceSquaredA * 0.5 );
            falloffBaseStemB = falloffBase - float32v( kDistanceSquaredB * 0.5 );
            value = ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ) * gradientRampValue;
            addVertex( hash, gradientRampValue, falloffBase, dxBase, dyBase, dzBase );
        }

        // Vertex <1, 1, 1> or <-1, -1, -1>
        {
            mask32v signMask = xyzNormal < float32v( 0 );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset = float32v( 3 * kReflectUnskew3 + 1 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimed, yPrimed, zPrimed );
            float32v gradientRampValue = GetGradientDotCommon( hash, dxBase - offset, dyBase - offset, dzBase - offset );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset, coordinateSum, falloffBaseStemA ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase - offset, dyBase - offset, dzBase - offset );
        }

        // Vertex <1, 1, 0> or <-1, -1, 0>
        {
            mask32v signMask = xyzNormal < zNormal;

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset0 = float32v( 2 * kReflectUnskew3 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimed, yPrimed, zPrimedBase );
            float32v gradientRampValue = GetGradientDotCommon( hash, dxBase, dyBase, dzBase - offset0 );
            float32v falloffBase = FS::Min( ( sign ^ dzBase ) - falloffBaseStemB, float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase, dyBase, dzBase - offset0 );
        }

        // Vertex <1, 0, 1> or <-1, 0, -1>
        {
            mask32v signMask = xyzNormal < yNormal;

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset0 = float32v( 2 * kReflectUnskew3 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimed, yPrimedBase, zPrimed );
            float32v gradientRampValue = GetGradientDotCommon( hash, dxBase, dyBase - offset0, dzBase );
            float32v falloffBase = FS::Min( ( sign ^ dyBase ) - falloffBaseStemB, float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase, dyBase - offset0, dzBase );
        }

        // Vertex <0, 1, 1> or <0, -1, -1>
        {
            mask32v signMask = xyzNormal < xNormal;

            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset0 = float32v( 2 * kReflectUnskew3 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimedBase, yPrimed, zPrimed );
            float32v gradientRampValue = GetGradientDotCommon( hash, dxBase - offset0, dyBase, dzBase );
            float32v falloffBase = FS::Min( ( sign ^ dxBase ) - falloffBaseStemB, float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase - offset0, dyBase, dzBase );
        }

        // Vertex <1, 0, 0> or <-1, 0, 0>
        {
            mask32v signMask = xNormal < float32v( 0 );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset0 = float32v( kReflectUnskew3 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimed, yPrimedBase, zPrimedBase );
            float32v gradientRampValue = GetGradientDotCommon( hash, dxBase + offset0, dyBase - offset0, dzBase - offset0 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemA ) + ( sign ^ dxBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase + offset0, dyBase - offset0, dzBase - offset0 );
        }

        // Vertex <0, 1, 0> or <0, -1, 0>
        {
            mask32v signMask = yNormal < float32v( 0 );

            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset0 = float32v( kReflectUnskew3 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimedBase, yPrimed, zPrimedBase );
            float32v gradientRampValue = GetGradientDotCommon( hash, dxBase - offset0, dyBase + offset0, dzBase - offset0 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemA ) + ( sign ^ dyBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase - offset0, dyBase + offset0, dzBase - offset0 );
        }

        // Vertex <0, 0, 1> or <0, 0, -1>
        {
            mask32v signMask = zNormal < float32v( 0 );

            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );
            float32v offset0 = float32v( kReflectUnskew3 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimed );
            float32v gradientRampValue = GetGradientDotCommon( hash, dxBase - offset0, dyBase - offset0, dzBase + offset0 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemA ) + ( sign ^ dzBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase - offset0, dyBase - offset0, dzBase + offset0 );
        }

        // The skew then reflected unskew is the reflection I - 2/3 * 11^T, which is its own transpose
        float32v reflect = float32v( -2.0 / 3.0 ) * ( gX + gY + gZ );
        gradX = gX + reflect;
        gradY = gY + reflect;
        gradZ = gZ + reflect;

        return value;
    }

    bool FS_VECTORCALL GenGradient( int32v seed, float32v x, float32v y, float32v& value, float32v& dx, float32v& dy ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y );

        value = ValueGradient( seed, x, y, dx, dy );

        float32v slope = this->ScaleOutputSlope( -kBounding2D, kBounding2D );
        dx *= slope;
        dy *= slope;
        value = this->ScaleOutput( value, -kBounding2D, kBounding2D );
        return true;
    }

    bool FS_VECTORCALL GenGradient( int32v seed, float32v x, float32v y, float32v z, float32v& value, float32v& dx, float32v& dy, float32v& dz ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y, z );

        value = ValueGradient( seed, x, y, z, dx, dy, dz );

        float32v slope = this->ScaleOutputSlope( -kBounding3D, kBounding3D );
        dx *= slope;
        dy *= slope;
        dz *= slope;
        value = this->ScaleOutput( value, -kBounding3D, kBounding3D );
        return true;
    }

    static constexpr float kBounding4D = 0.011310013942420482635498046875f;

    static FS_FORCEINLINE float32v ValueGradient( int32v seed, float32v x, float32v y, float32v z, float32v w, float32v& gradX, float32v& gradY, float32v& gradZ, float32v& gradW )
    {
        constexpr double kRoot5 = 2.2360679774997896964091736687313;
        constexpr double kSkew4 = 1.0 / ( kRoot5 + 1.0 );
        constexpr double kUnskew4 = -1.0 / ( kRoot5 + 5.0 );
        constexpr double kTwiceUnskew4 = -1.0 / 5.0;

        constexpr double kDistanceSquaredA = 4.0 / 5.0;
        constexpr double kDistanceSquaredB = 6.0 / 5.0;
        constexpr double kFalloffRadiusSquared = kDistanceSquaredA;

        float32v skewDelta = float32v( kSkew4 ) * ( x + y + z + w );

        float32v xSkewed = x + skewDelta;
        float32v ySkewed = y + skewDelta;
        float32v zSkewed = z + skewDelta;
        float32v wSkewed = w + skewDelta;
        float32v xSkewedBase = FS::Floor( xSkewed );
        float32v ySkewedBase = FS::Floor( ySkewed );
        float32v zSkewedBase = FS::Floor( zSkewed );
        float32v wSkewedBase = FS::Floor( wSkewed );
        float32v dxSkewed = xSkewed - xSkewedBase;
        float32v dySkewed = ySkewed - ySkewedBase;
        float32v dzSkewed = zSkewed - zSkewedBase;
        float32v dwSkewed = wSkewed - wSkewedBase;

        // From unit cell base, find closest vertex
        {
            float32v twiceUnskewDelta = float32v( kTwiceUnskew4 ) * ( dxSkewed + dySkewed + dzSkewed + dwSkewed );
            float32v xNormal = dxSkewed + twiceUnskewDelta;
            float32v yNormal = dySkewed + twiceUnskewDelta;
            float32v zNormal = dzSkewed + twiceUnskewDelta;
            float32v wNormal = dwSkewed + twiceUnskewDelta;
            float32v xyzwNormal = -twiceUnskewDelta;

            constexpr auto considerVertex = []( float32v& maxScore, int32v& moveMaskBits, float32v score, int32v bits ) constexpr {
                moveMaskBits = FS::Select( score > maxScore, bits, moveMaskBits );
                maxScore = FS::Max( maxScore, score );
            };
            float32v maxScore = float32v( 0.6f ) - xyzwNormal;
            int32v moveMaskBits = FS::Masked( float32v( 0.2f ) > maxScore, int32v( -1 ) );
            maxScore = FS::Max( maxScore, float32v( 0.2f ) );
            considerVertex( maxScore, moveMaskBits, -wNormal, 0b0111 );
            considerVertex( maxScore, moveMaskBits, -zNormal, 0b1011 );
            considerVertex( maxScore, moveMaskBits, -yNormal, 0b1101 );
            considerVertex( maxScore, moveMaskBits, -xNormal, 0b1110 );
            maxScore += xyzwNormal - float32v( 0.2f );
            considerVertex( maxScore, moveMaskBits, xNormal, 0b0001 );
            considerVertex( maxScore, moveMaskBits, yNormal, 0b0010 );
            considerVertex( maxScore, moveMaskBits, zNormal, 0b0100 );
            considerVertex( maxScore, moveMaskBits, wNormal, 0b1000 );
            maxScore += float32v( 0.2f ) - xNormal;
            considerVertex( maxScore, moveMaskBits, yNormal, 0b0011 );
            considerVertex( maxScore, moveMaskBits, zNormal, 0b0101 );
            considerVertex( maxScore, moveMaskBits, wNormal, 0b1001 );
            maxScore += xNormal;
            considerVertex( maxScore, moveMaskBits, yNormal + zNormal, 0b0110 );
            maxScore -= wNormal;
            considerVertex( maxScore, moveMaskBits, yNormal, 0b1010 );
            considerVertex( maxScore, moveMaskBits, zNormal, 0b1100 );

            mask32v moveX = ( moveMaskBits & int32v( 0b0001 ) ) != int32v( 0 );
            mask32v moveY = ( moveMaskBits & int32v( 0b0010 ) ) != int32v( 0 );
            mask32v moveZ = ( moveMaskBits & int32v( 0b0100 ) ) != int32v( 0 );
            mask32v moveW = ( moveMaskBits & int32v( 0b1000 ) ) != int32v( 0 );

            xSkewedBase = FS::MaskedIncrement( moveX, xSkewedBase );
            ySkewedBase = FS::MaskedIncrement( moveY, ySkewedBase );
            zSkewedBase = FS::MaskedIncrement( moveZ, zSkewedBase );
            wSkewedBase = FS::MaskedIncrement( moveW, wSkewedBase );

            dxSkewed = FS::MaskedDecrement( moveX, dxSkewed );
            dySkewed = FS::MaskedDecrement( moveY, dySkewed );
            dzSkewed = FS::MaskedDecrement( moveZ, dzSkewed );
            dwSkewed = FS::MaskedDecrement( moveW, dwSkewed );
        }

        int32v xPrimedBase = FS::Convert<int32_t>( xSkewedBase ) * int32v( Primes::X );
        int32v yPrimedBase = FS::Convert<int32_t>( ySkewedBase ) * int32v( Primes::Y );
        int32v zPrimedBase = FS::Convert<int32_t>( zSkewedBase ) * int32v( Primes::Z );
        int32v wPrimedBase = FS::Convert<int32_t>( wSkewedBase ) * int32v( Primes::W );

        float32v skewedCoordinateSum = dxSkewed + dySkewed + dzSkewed + dwSkewed;
        float32v twiceUnskewDelta = float32v( kTwiceUnskew4 ) * skewedCoordinateSum;
        float32v xNormal = dxSkewed + twiceUnskewDelta;
        float32v yNormal = dySkewed + twiceUnskewDelta;
        float32v zNormal = dzSkewed + twiceUnskewDelta;
        float32v wNormal = dwSkewed + twiceUnskewDelta;
        float32v xyzwNormal = -twiceUnskewDelta;

        float32v unskewDelta = float32v( kUnskew4 ) * skewedCoordinateSum;
        float32v dxBase = dxSkewed + unskewDelta;
        float32v dyBase = dySkewed + unskewDelta;
        float32v dzBase = dzSkewed + unskewDelta;
        float32v dwBase = dwSkewed + unskewDelta;

        float32v coordinateSum = float32v( 1 + 4 * kUnskew4 ) * skewedCoordinateSum;

        // Each falloff is +-(r^2 - |d|^2) / 2, so f^4 * g.d contributes f^4 * g - 4 * |f|^3 * (g.d) * d; the 4D unskew Jacobian is identity
        gradX = gradY = gradZ = gradW = float32v( 0 );
        auto addVertex = [&]( int32v hash, float32v ramp, float32v falloffBase, float32v dx, float32v dy, float32v dz, float32v dw ) {
            float32v f = FS::Abs( falloffBase );
            float32v f2 = f * f;
            float32v radial = float32v( -4 ) * f2 * f * ramp;
            float32v gx, gy, gz, gw;
            GetGradientPerlin( hash, gx, gy, gz, gw );
            gradX += FS::FMulAdd( f2 * f2, gx, radial * dx );
            gradY += FS::FMulAdd( f2 * f2, gy, radial * dy );
            gradZ += FS::FMulAdd( f2 * f2, gz, radial * dz );
            gradW += FS::FMulAdd( f2 * f2, gw, radial * dw );
        };

        // Vertex <0, 0, 0, 0>
        float32v value, falloffBaseStemA, falloffBaseStemB;
        {
            int32v hash = HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimedBase, wPrimedBase );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dxBase, dyBase, dzBase, dwBase );
            float32v falloffBase = FS::FNMulAdd( dwBase, dwBase, FS::FNMulAdd( dzBase, dzBase, FS::FNMulAdd( dyBase, dyBase, FS::FNMulAdd( dxBase, dxBase, float32v( kFalloffRadiusSquared ) ) ) ) ) * float32v( 0.5f );
            falloffBaseStemA = falloffBase - float32v( kDistanceSquaredA * 0.5 );
            falloffBaseStemB = falloffBase - float32v( kDistanceSquaredB * 0.5 );
            value = ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ) * gradientRampValue;
            addVertex( hash, gradientRampValue, falloffBase, dxBase, dyBase, dzBase, dwBase );
        }

        // Vertex <1, 1, 1, 1> or <-1, -1, -1, -1>
        {
            mask32v signMask = xyzwNormal < float32v( 0 );
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset = float32v( 4 * kUnskew4 + 1 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimed, yPrimed, zPrimed, wPrimed );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dxBase - offset, dyBase - offset, dzBase - offset, dwBase - offset );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset, coordinateSum, falloffBaseStemA ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase - offset, dyBase - offset, dzBase - offset, dwBase - offset );
        }

        // Vertex <1, 1, 1, 0> or <-1, -1, -1, 0>
        {
            mask32v signMask = xyzwNormal < wNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v offset1 = float32v( 3 * kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( 3 * kUnskew4 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimed, yPrimed, zPrimed, wPrimedBase );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dxBase - offset1, dyBase - offset1, dzBase - offset1, dwBase - offset0 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset1, coordinateSum, falloffBaseStemB ) - ( sign ^ dwBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase - offset1, dyBase - offset1, dzBase - offset1, dwBase - offset0 );
        }

        // Vertex <1, 1, 0, 1> or <-1, -1, 0, -1>
        {
            mask32v signMask = xyzwNormal < zNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( 3 * kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( 3 * kUnskew4 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimed, yPrimed, zPrimedBase, wPrimed );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dxBase - offset1, dyBase - offset1, dzBase - offset0, dwBase - offset1 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset1, coordinateSum, falloffBaseStemB ) - ( sign ^ dzBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase - offset1, dyBase - offset1, dzBase - offset0, dwBase - offset1 );
        }

        // Vertex <1, 0, 1, 1> or <-1, 0, -1, -1>
        {
            mask32v signMask = xyzwNormal < yNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( 3 * kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( 3 * kUnskew4 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimed, yPrimedBase, zPrimed, wPrimed );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dxBase - offset1, dyBase - offset0, dzBase - offset1, dwBase - offset1 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset1, coordinateSum, falloffBaseStemB ) - ( sign ^ dyBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase - offset1, dyBase - offset0, dzBase - offset1, dwBase - offset1 );
        }

        // Vertex <0, 1, 1, 1> or <0, -1, -1, -1>
        {
            mask32v signMask = xyzwNormal < xNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( 3 * kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( 3 * kUnskew4 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimedBase, yPrimed, zPrimed, wPrimed );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dxBase - offset0, dyBase - offset1, dzBase - offset1, dwBase - offset1 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset1, coordinateSum, falloffBaseStemB ) - ( sign ^ dxBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase - offset0, dyBase - offset1, dzBase - offset1, dwBase - offset1 );
        }

        // Vertex <1, 0, 0, 0> or <-1, 0, 0, 0>
        {
            mask32v signMask = xNormal < float32v( 0 );
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );

            float32v offset1 = float32v( kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( kUnskew4 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimed, yPrimedBase, zPrimedBase, wPrimedBase );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dxBase - offset1, dyBase - offset0, dzBase - offset0, dwBase - offset0 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemA ) + ( sign ^ dxBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase - offset1, dyBase - offset0, dzBase - offset0, dwBase - offset0 );
        }

        // Vertex <1, 1, 0, 0> or <-1, -1, 0, 0>
        {
            mask32v signMask = xNormal < -yNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );

            float32v offset1 = float32v( 2 * kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( 2 * kUnskew4 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimed, yPrimed, zPrimedBase, wPrimedBase );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dxBase - offset1, dyBase - offset1, dzBase - offset0, dwBase - offset0 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemB ) + ( sign ^ ( dxBase + dyBase ) ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase - offset1, dyBase - offset1, dzBase - offset0, dwBase - offset0 );
        }

        // Vertex <1, 0, 1, 0> or <-1, 0, -1, 0>
        {
            mask32v signMask = xNormal < -zNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v offset1 = float32v( 2 * kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( 2 * kUnskew4 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimed, yPrimedBase, zPrimed, wPrimedBase );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dxBase - offset1, dyBase - offset0, dzBase - offset1, dwBase - offset0 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemB ) + ( sign ^ ( dxBase + dzBase ) ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase - offset1, dyBase - offset0, dzBase - offset1, dwBase - offset0 );
        }

        // Vertex <1, 0, 0, 1> or <-1, 0, 0, -1>
        {
            mask32v signMask = xNormal < -wNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v xPrimed = xPrimedBase + FS::Select( signMask, int32v( -Primes::X ), int32v( Primes::X ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( 2 * kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( 2 * kUnskew4 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimed, yPrimedBase, zPrimedBase, wPrimed );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dxBase - offset1, dyBase - offset0, dzBase - offset0, dwBase - offset1 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemB ) + ( sign ^ ( dxBase + dwBase ) ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase - offset1, dyBase - offset0, dzBase - offset0, dwBase - offset1 );
        }

        // Vertex <0, 1, 0, 0> or <0, -1, 0, 0>
        {
            mask32v signMask = yNormal < float32v( 0 );
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );

            float32v offset1 = float32v( kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( kUnskew4 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimedBase, yPrimed, zPrimedBase, wPrimedBase );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dxBase - offset0, dyBase - offset1, dzBase - offset0, dwBase - offset0 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemA ) + ( sign ^ dyBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase - offset0, dyBase - offset1, dzBase - offset0, dwBase - offset0 );
        }

        // Vertex <0, 1, 1, 0> or <0, -1, -1, 0>
        {
            mask32v signMask = yNormal < -zNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v offset1 = float32v( 2 * kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( 2 * kUnskew4 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimedBase, yPrimed, zPrimed, wPrimedBase );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dxBase - offset0, dyBase - offset1, dzBase - offset1, dwBase - offset0 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemB ) + ( sign ^ ( dyBase + dzBase ) ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase - offset0, dyBase - offset1, dzBase - offset1, dwBase - offset0 );
        }

        // Vertex <0, 1, 0, 1> or <0, -1, 0, -1>
        {
            mask32v signMask = yNormal < -wNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v yPrimed = yPrimedBase + FS::Select( signMask, int32v( -Primes::Y ), int32v( Primes::Y ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( 2 * kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( 2 * kUnskew4 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimedBase, yPrimed, zPrimedBase, wPrimed );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dxBase - offset0, dyBase - offset1, dzBase - offset0, dwBase - offset1 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemB ) + ( sign ^ ( dyBase + dwBase ) ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase - offset0, dyBase - offset1, dzBase - offset0, dwBase - offset1 );
        }

        // Vertex <0, 0, 1, 0> or <0, 0, -1, 0>
        {
            mask32v signMask = zNormal < float32v( 0 );
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );

            float32v offset1 = float32v( kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( kUnskew4 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimed, wPrimedBase );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dxBase - offset0, dyBase - offset0, dzBase - offset1, dwBase - offset0 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemA ) + ( sign ^ dzBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase - offset0, dyBase - offset0, dzBase - offset1, dwBase - offset0 );
        }

        // Vertex <0, 0, 1, 1> or <0, 0, -1, -1>
        {
            mask32v signMask = zNormal < -wNormal;
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v zPrimed = zPrimedBase + FS::Select( signMask, int32v( -Primes::Z ), int32v( Primes::Z ) );
            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( 2 * kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( 2 * kUnskew4 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimed, wPrimed );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dxBase - offset0, dyBase - offset0, dzBase - offset1, dwBase - offset1 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemB ) + ( sign ^ ( dzBase + dwBase ) ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase - offset0, dyBase - offset0, dzBase - offset1, dwBase - offset1 );
        }

        // Vertex <0, 0, 0, 1> or <0, 0, 0, -1>
        {
            mask32v signMask = wNormal < float32v( 0 );
            float32v sign = FS::Masked( signMask, float32v( FS::Cast<float>( int32v( 1 << 31 ) ) ) );

            int32v wPrimed = wPrimedBase + FS::Select( signMask, int32v( -Primes::W ), int32v( Primes::W ) );

            float32v offset1 = float32v( kUnskew4 + 1 ) ^ sign;
            float32v offset0 = float32v( kUnskew4 ) ^ sign;

            int32v hash = HashPrimes( seed, xPrimedBase, yPrimedBase, zPrimedBase, wPrimed );
            float32v gradientRampValue = GetGradientDotPerlin( hash, dxBase - offset0, dyBase - offset0, dzBase - offset0, dwBase - offset1 );
            float32v falloffBase = FS::Max( FS::FMulAdd( offset0, coordinateSum, falloffBaseStemA ) + ( sign ^ dwBase ), float32v( 0.0f ) );
            value = FS::FMulAdd( ( falloffBase * falloffBase ) * ( falloffBase * falloffBase ), gradientRampValue, value );
            addVertex( hash, gradientRampValue, falloffBase, dxBase - offset0, dyBase - offset0, dzBase - offset0, dwBase - offset1 );
        }

        return value;
    }

    bool FS_VECTORCALL GenGradient( int32v seed, float32v x, float32v y, float32v z, float32v w, float32v& value, float32v& dx, float32v& dy, float32v& dz, float32v& dw ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y, z, w );

        value = ValueGradient( seed, x, y, z, w, dx, dy, dz, dw );

        float32v slope = this->ScaleOutputSlope( -kBounding4D, kBounding4D );
        dx *= slope;
        dy *= slope;
        dz *= slope;
        dw *= slope;
        value = this->ScaleOutput( value, -kBounding4D, kBounding4D );
        return true;
    }
};
