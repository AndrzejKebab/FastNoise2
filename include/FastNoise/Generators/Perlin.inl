#include "Perlin.h"
#include "Utils.inl"

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<Perlin, SIMD> final : public virtual Perlin, public DispatchClass<VariableRange<Seeded<ScalableGenerator>>, SIMD>
{
    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y );

        float32v xs = FS::Floor( x );
        float32v ys = FS::Floor( y );

        int32v x0 = FS::Convert<int32_t>( xs ) * int32v( Primes::X );
        int32v y0 = FS::Convert<int32_t>( ys ) * int32v( Primes::Y );
        int32v x1 = x0 + int32v( Primes::X );
        int32v y1 = y0 + int32v( Primes::Y );

        float32v xf0 = xs = x - xs;
        float32v yf0 = ys = y - ys;
        float32v xf1 = xf0 - float32v( 1 );
        float32v yf1 = yf0 - float32v( 1 );

        xs = InterpQuintic( xs );
        ys = InterpQuintic( ys );

        float32v value = Lerp(
            Lerp( GetGradientDotPerlin( HashPrimes( seed, x0, y0 ), xf0, yf0 ), GetGradientDotPerlin( HashPrimes( seed, x1, y0 ), xf1, yf0 ), xs ),
            Lerp( GetGradientDotPerlin( HashPrimes( seed, x0, y1 ), xf0, yf1 ), GetGradientDotPerlin( HashPrimes( seed, x1, y1 ), xf1, yf1 ), xs ), ys );

        constexpr float kBounding = 1.726796627044677734375f;

        return this->ScaleOutput( value, -kBounding, kBounding );
    }

    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y, z );

        float32v xs = FS::Floor( x );
        float32v ys = FS::Floor( y );
        float32v zs = FS::Floor( z );

        int32v x0 = FS::Convert<int32_t>( xs ) * int32v( Primes::X );
        int32v y0 = FS::Convert<int32_t>( ys ) * int32v( Primes::Y );
        int32v z0 = FS::Convert<int32_t>( zs ) * int32v( Primes::Z );
        int32v x1 = x0 + int32v( Primes::X );
        int32v y1 = y0 + int32v( Primes::Y );
        int32v z1 = z0 + int32v( Primes::Z );

        float32v xf0 = xs = x - xs;
        float32v yf0 = ys = y - ys;
        float32v zf0 = zs = z - zs;
        float32v xf1 = xf0 - float32v( 1 );
        float32v yf1 = yf0 - float32v( 1 );
        float32v zf1 = zf0 - float32v( 1 );

        xs = InterpQuintic( xs );
        ys = InterpQuintic( ys );
        zs = InterpQuintic( zs );

        float32v value = Lerp( Lerp(
            Lerp( GetGradientDotCommon( HashPrimes( seed, x0, y0, z0 ), xf0, yf0, zf0 ), GetGradientDotCommon( HashPrimes( seed, x1, y0, z0 ), xf1, yf0, zf0 ), xs ),
            Lerp( GetGradientDotCommon( HashPrimes( seed, x0, y1, z0 ), xf0, yf1, zf0 ), GetGradientDotCommon( HashPrimes( seed, x1, y1, z0 ), xf1, yf1, zf0 ), xs ), ys ),
            Lerp(
            Lerp( GetGradientDotCommon( HashPrimes( seed, x0, y0, z1 ), xf0, yf0, zf1 ), GetGradientDotCommon( HashPrimes( seed, x1, y0, z1 ), xf1, yf0, zf1 ), xs ),
            Lerp( GetGradientDotCommon( HashPrimes( seed, x0, y1, z1 ), xf0, yf1, zf1 ), GetGradientDotCommon( HashPrimes( seed, x1, y1, z1 ), xf1, yf1, zf1 ), xs ), ys ), zs );

        constexpr double kBounding = 1.0363423824310302734375;

        return this->ScaleOutput( value, -kBounding, kBounding );
    }

    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z, float32v w ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y, z, w );

        float32v xs = FS::Floor( x );
        float32v ys = FS::Floor( y );
        float32v zs = FS::Floor( z );
        float32v ws = FS::Floor( w );

        int32v x0 = FS::Convert<int32_t>( xs ) * int32v( Primes::X );
        int32v y0 = FS::Convert<int32_t>( ys ) * int32v( Primes::Y );
        int32v z0 = FS::Convert<int32_t>( zs ) * int32v( Primes::Z );
        int32v w0 = FS::Convert<int32_t>( ws ) * int32v( Primes::W );
        int32v x1 = x0 + int32v( Primes::X );
        int32v y1 = y0 + int32v( Primes::Y );
        int32v z1 = z0 + int32v( Primes::Z );
        int32v w1 = w0 + int32v( Primes::W );

        float32v xf0 = xs = x - xs;
        float32v yf0 = ys = y - ys;
        float32v zf0 = zs = z - zs;
        float32v wf0 = ws = w - ws;
        float32v xf1 = xf0 - float32v( 1 );
        float32v yf1 = yf0 - float32v( 1 );
        float32v zf1 = zf0 - float32v( 1 );
        float32v wf1 = wf0 - float32v( 1 );

        xs = InterpQuintic( xs );
        ys = InterpQuintic( ys );
        zs = InterpQuintic( zs );
        ws = InterpQuintic( ws );

        float32v value = Lerp( Lerp( Lerp(
            Lerp( GetGradientDotPerlin( HashPrimes( seed, x0, y0, z0, w0 ), xf0, yf0, zf0, wf0 ), GetGradientDotPerlin( HashPrimes( seed, x1, y0, z0, w0 ), xf1, yf0, zf0, wf0 ), xs ),
            Lerp( GetGradientDotPerlin( HashPrimes( seed, x0, y1, z0, w0 ), xf0, yf1, zf0, wf0 ), GetGradientDotPerlin( HashPrimes( seed, x1, y1, z0, w0 ), xf1, yf1, zf0, wf0 ), xs ), ys ),
            Lerp(
            Lerp( GetGradientDotPerlin( HashPrimes( seed, x0, y0, z1, w0 ), xf0, yf0, zf1, wf0 ), GetGradientDotPerlin( HashPrimes( seed, x1, y0, z1, w0 ), xf1, yf0, zf1, wf0 ), xs ),
            Lerp( GetGradientDotPerlin( HashPrimes( seed, x0, y1, z1, w0 ), xf0, yf1, zf1, wf0 ), GetGradientDotPerlin( HashPrimes( seed, x1, y1, z1, w0 ), xf1, yf1, zf1, wf0 ), xs ), ys ), zs ),
            Lerp( Lerp(
            Lerp( GetGradientDotPerlin( HashPrimes( seed, x0, y0, z0, w1 ), xf0, yf0, zf0, wf1 ), GetGradientDotPerlin( HashPrimes( seed, x1, y0, z0, w1 ), xf1, yf0, zf0, wf1 ), xs ),
            Lerp( GetGradientDotPerlin( HashPrimes( seed, x0, y1, z0, w1 ), xf0, yf1, zf0, wf1 ), GetGradientDotPerlin( HashPrimes( seed, x1, y1, z0, w1 ), xf1, yf1, zf0, wf1 ), xs ), ys ),
            Lerp(
            Lerp( GetGradientDotPerlin( HashPrimes( seed, x0, y0, z1, w1 ), xf0, yf0, zf1, wf1 ), GetGradientDotPerlin( HashPrimes( seed, x1, y0, z1, w1 ), xf1, yf0, zf1, wf1 ), xs ),
            Lerp( GetGradientDotPerlin( HashPrimes( seed, x0, y1, z1, w1 ), xf0, yf1, zf1, wf1 ), GetGradientDotPerlin( HashPrimes( seed, x1, y1, z1, w1 ), xf1, yf1, zf1, wf1 ), xs ), ys ) , zs ), ws );

        constexpr float kBounding = 1.33858621120452880859375;

        return this->ScaleOutput( value, -kBounding, kBounding );
    }

public:
    static constexpr float kBounding2D = 1.726796627044677734375f;
    static constexpr float kBounding3D = 1.0363423824310302734375f;
    static constexpr float kBounding4D = 1.33858621120452880859375f;

    static FS_FORCEINLINE float32v InterpQuinticDerivative( float32v t )
    {
        return float32v( 30 ) * t * t * FS::FMulAdd( t, t - float32v( 2 ), float32v( 1 ) );
    }

    // Same arithmetic as Gen after ScalePositions, returns the native value and its gradient per noise cell
    static FS_FORCEINLINE float32v ValueGradient( int32v seed, float32v x, float32v y, float32v& gradX, float32v& gradY )
    {
        float32v xs = FS::Floor( x );
        float32v ys = FS::Floor( y );

        int32v x0 = FS::Convert<int32_t>( xs ) * int32v( Primes::X );
        int32v y0 = FS::Convert<int32_t>( ys ) * int32v( Primes::Y );
        int32v x1 = x0 + int32v( Primes::X );
        int32v y1 = y0 + int32v( Primes::Y );

        float32v xf0 = xs = x - xs;
        float32v yf0 = ys = y - ys;
        float32v xf1 = xf0 - float32v( 1 );
        float32v yf1 = yf0 - float32v( 1 );

        float32v xd = InterpQuinticDerivative( xs );
        float32v yd = InterpQuinticDerivative( ys );
        xs = InterpQuintic( xs );
        ys = InterpQuintic( ys );

        int32v h00 = HashPrimes( seed, x0, y0 );
        int32v h10 = HashPrimes( seed, x1, y0 );
        int32v h01 = HashPrimes( seed, x0, y1 );
        int32v h11 = HashPrimes( seed, x1, y1 );

        float32v v00 = GetGradientDotPerlin( h00, xf0, yf0 );
        float32v v10 = GetGradientDotPerlin( h10, xf1, yf0 );
        float32v v01 = GetGradientDotPerlin( h01, xf0, yf1 );
        float32v v11 = GetGradientDotPerlin( h11, xf1, yf1 );

        float32v value = Lerp( Lerp( v00, v10, xs ), Lerp( v01, v11, xs ), ys );

        // Interpolated corner gradients, plus the interpolant's slope times the difference across the cell
        int32v hash[4] = { h00, h10, h01, h11 };
        float32v gx[4], gy[4];
        for( int i = 0; i < 4; i++ )
        {
            GetGradientPerlin( hash[i], gx[i], gy[i] );
        }
        gradX = FS::FMulAdd( xd, Lerp( v10, v11, ys ) - Lerp( v00, v01, ys ), Lerp( Lerp( gx[0], gx[1], xs ), Lerp( gx[2], gx[3], xs ), ys ) );
        gradY = FS::FMulAdd( yd, Lerp( v01, v11, xs ) - Lerp( v00, v10, xs ), Lerp( Lerp( gy[0], gy[1], xs ), Lerp( gy[2], gy[3], xs ), ys ) );

        return value;
    }

    static FS_FORCEINLINE float32v ValueGradient( int32v seed, float32v x, float32v y, float32v z, float32v& gradX, float32v& gradY, float32v& gradZ )
    {
        float32v xs = FS::Floor( x );
        float32v ys = FS::Floor( y );
        float32v zs = FS::Floor( z );

        int32v x0 = FS::Convert<int32_t>( xs ) * int32v( Primes::X );
        int32v y0 = FS::Convert<int32_t>( ys ) * int32v( Primes::Y );
        int32v z0 = FS::Convert<int32_t>( zs ) * int32v( Primes::Z );
        int32v x1 = x0 + int32v( Primes::X );
        int32v y1 = y0 + int32v( Primes::Y );
        int32v z1 = z0 + int32v( Primes::Z );

        float32v xf0 = xs = x - xs;
        float32v yf0 = ys = y - ys;
        float32v zf0 = zs = z - zs;
        float32v xf1 = xf0 - float32v( 1 );
        float32v yf1 = yf0 - float32v( 1 );
        float32v zf1 = zf0 - float32v( 1 );

        float32v xd = InterpQuinticDerivative( xs );
        float32v yd = InterpQuinticDerivative( ys );
        float32v zd = InterpQuinticDerivative( zs );
        xs = InterpQuintic( xs );
        ys = InterpQuintic( ys );
        zs = InterpQuintic( zs );

        int32v h000 = HashPrimes( seed, x0, y0, z0 );
        int32v h100 = HashPrimes( seed, x1, y0, z0 );
        int32v h010 = HashPrimes( seed, x0, y1, z0 );
        int32v h110 = HashPrimes( seed, x1, y1, z0 );
        int32v h001 = HashPrimes( seed, x0, y0, z1 );
        int32v h101 = HashPrimes( seed, x1, y0, z1 );
        int32v h011 = HashPrimes( seed, x0, y1, z1 );
        int32v h111 = HashPrimes( seed, x1, y1, z1 );

        float32v v000 = GetGradientDotCommon( h000, xf0, yf0, zf0 );
        float32v v100 = GetGradientDotCommon( h100, xf1, yf0, zf0 );
        float32v v010 = GetGradientDotCommon( h010, xf0, yf1, zf0 );
        float32v v110 = GetGradientDotCommon( h110, xf1, yf1, zf0 );
        float32v v001 = GetGradientDotCommon( h001, xf0, yf0, zf1 );
        float32v v101 = GetGradientDotCommon( h101, xf1, yf0, zf1 );
        float32v v011 = GetGradientDotCommon( h011, xf0, yf1, zf1 );
        float32v v111 = GetGradientDotCommon( h111, xf1, yf1, zf1 );

        float32v value = Lerp( Lerp(
                                   Lerp( v000, v100, xs ),
                                   Lerp( v010, v110, xs ), ys ),
                               Lerp(
                                   Lerp( v001, v101, xs ),
                                   Lerp( v011, v111, xs ), ys ),
                               zs );

        int32v hash[8] = { h000, h100, h010, h110, h001, h101, h011, h111 };
        float32v g[3][8];
        for( int i = 0; i < 8; i++ )
        {
            GetGradientCommon( hash[i], g[0][i], g[1][i], g[2][i] );
        }
        auto trilinear = [&]( const float32v( &c )[8] ) {
            return Lerp( Lerp( Lerp( c[0], c[1], xs ), Lerp( c[2], c[3], xs ), ys ), Lerp( Lerp( c[4], c[5], xs ), Lerp( c[6], c[7], xs ), ys ), zs );
        };
        gradX = FS::FMulAdd( xd,
                             Lerp( Lerp( v100, v110, ys ), Lerp( v101, v111, ys ), zs ) - Lerp( Lerp( v000, v010, ys ), Lerp( v001, v011, ys ), zs ),
                             trilinear( g[0] ) );
        gradY = FS::FMulAdd( yd,
                             Lerp( Lerp( v010, v110, xs ), Lerp( v011, v111, xs ), zs ) - Lerp( Lerp( v000, v100, xs ), Lerp( v001, v101, xs ), zs ),
                             trilinear( g[1] ) );
        gradZ = FS::FMulAdd( zd,
                             Lerp( Lerp( v001, v101, xs ), Lerp( v011, v111, xs ), ys ) - Lerp( Lerp( v000, v100, xs ), Lerp( v010, v110, xs ), ys ),
                             trilinear( g[2] ) );

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

    static FS_FORCEINLINE float32v ValueGradient( int32v seed, float32v x, float32v y, float32v z, float32v w, float32v& gradX, float32v& gradY, float32v& gradZ, float32v& gradW )
    {
        float32v xs = FS::Floor( x );
        float32v ys = FS::Floor( y );
        float32v zs = FS::Floor( z );
        float32v ws = FS::Floor( w );

        int32v x0 = FS::Convert<int32_t>( xs ) * int32v( Primes::X );
        int32v y0 = FS::Convert<int32_t>( ys ) * int32v( Primes::Y );
        int32v z0 = FS::Convert<int32_t>( zs ) * int32v( Primes::Z );
        int32v w0 = FS::Convert<int32_t>( ws ) * int32v( Primes::W );
        int32v x1 = x0 + int32v( Primes::X );
        int32v y1 = y0 + int32v( Primes::Y );
        int32v z1 = z0 + int32v( Primes::Z );
        int32v w1 = w0 + int32v( Primes::W );

        float32v xf0 = xs = x - xs;
        float32v yf0 = ys = y - ys;
        float32v zf0 = zs = z - zs;
        float32v wf0 = ws = w - ws;
        float32v xf1 = xf0 - float32v( 1 );
        float32v yf1 = yf0 - float32v( 1 );
        float32v zf1 = zf0 - float32v( 1 );
        float32v wf1 = wf0 - float32v( 1 );

        float32v derivative[4] = { InterpQuinticDerivative( xs ), InterpQuinticDerivative( ys ), InterpQuinticDerivative( zs ), InterpQuinticDerivative( ws ) };
        xs = InterpQuintic( xs );
        ys = InterpQuintic( ys );
        zs = InterpQuintic( zs );
        ws = InterpQuintic( ws );

        int32v hash[16];
        float32v corner[16];
        for( int i = 0; i < 16; i++ )
        {
            hash[i] = HashPrimes( seed, i & 1 ? x1 : x0, i & 2 ? y1 : y0, i & 4 ? z1 : z0, i & 8 ? w1 : w0 );
            corner[i] = GetGradientDotPerlin( hash[i], i & 1 ? xf1 : xf0, i & 2 ? yf1 : yf0, i & 4 ? zf1 : zf0, i & 8 ? wf1 : wf0 );
        }

        float32v value = Lerp4D( corner, xs, ys, zs, ws );

        // Interpolated corner gradients, plus the interpolant's slope times the difference across the cell
        float32v cornerGradient[4][16];
        for( int i = 0; i < 16; i++ )
        {
            GetGradientPerlin( hash[i], cornerGradient[0][i], cornerGradient[1][i], cornerGradient[2][i], cornerGradient[3][i] );
        }

        float32v* gradient[4] = { &gradX, &gradY, &gradZ, &gradW };
        for( int axis = 0; axis < 4; axis++ )
        {
            float32v across[16];
            for( int i = 0; i < 16; i++ )
            {
                across[i] = corner[i | ( 1 << axis )] - corner[i & ~( 1 << axis )];
            }
            *gradient[axis] = FS::FMulAdd( derivative[axis], Lerp4D( across, xs, ys, zs, ws ), Lerp4D( cornerGradient[axis], xs, ys, zs, ws ) );
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
