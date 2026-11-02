#include "Value.h"
#include "Utils.inl"

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<Value, SIMD> final : public virtual Value, public DispatchClass<VariableRange<Seeded<ScalableGenerator>>, SIMD>
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

        xs = InterpHermite( x - xs );
        ys = InterpHermite( y - ys );

        return this->ScaleOutput( Lerp(
            Lerp( GetValueCoord( seed, x0, y0 ), GetValueCoord( seed, x1, y0 ), xs ),
            Lerp( GetValueCoord( seed, x0, y1 ), GetValueCoord( seed, x1, y1 ), xs ), ys ),
            -kValueBounds, kValueBounds );
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

        xs = InterpHermite( x - xs );
        ys = InterpHermite( y - ys );
        zs = InterpHermite( z - zs );

        return this->ScaleOutput( Lerp( Lerp(
            Lerp( GetValueCoord( seed, x0, y0, z0 ), GetValueCoord( seed, x1, y0, z0 ), xs ),
            Lerp( GetValueCoord( seed, x0, y1, z0 ), GetValueCoord( seed, x1, y1, z0 ), xs ), ys ),
            Lerp(                                                                                
            Lerp( GetValueCoord( seed, x0, y0, z1 ), GetValueCoord( seed, x1, y0, z1 ), xs ),    
            Lerp( GetValueCoord( seed, x0, y1, z1 ), GetValueCoord( seed, x1, y1, z1 ), xs ), ys ), zs ),
            -kValueBounds, kValueBounds );
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

        xs = InterpHermite( x - xs );
        ys = InterpHermite( y - ys );
        zs = InterpHermite( z - zs );
        ws = InterpHermite( w - ws );

        return this->ScaleOutput( Lerp( Lerp( Lerp(
            Lerp( GetValueCoord( seed, x0, y0, z0, w0 ), GetValueCoord( seed, x1, y0, z0, w0 ), xs ),
            Lerp( GetValueCoord( seed, x0, y1, z0, w0 ), GetValueCoord( seed, x1, y1, z0, w0 ), xs ), ys ),
            Lerp( 
            Lerp( GetValueCoord( seed, x0, y0, z1, w0 ), GetValueCoord( seed, x1, y0, z1, w0 ), xs ),    
            Lerp( GetValueCoord( seed, x0, y1, z1, w0 ), GetValueCoord( seed, x1, y1, z1, w0 ), xs ), ys ), zs ),
            Lerp( Lerp(
            Lerp( GetValueCoord( seed, x0, y0, z0, w1 ), GetValueCoord( seed, x1, y0, z0, w1 ), xs ),
            Lerp( GetValueCoord( seed, x0, y1, z0, w1 ), GetValueCoord( seed, x1, y1, z0, w1 ), xs ), ys ),
            Lerp( 
            Lerp( GetValueCoord( seed, x0, y0, z1, w1 ), GetValueCoord( seed, x1, y0, z1, w1 ), xs ),    
            Lerp( GetValueCoord( seed, x0, y1, z1, w1 ), GetValueCoord( seed, x1, y1, z1, w1 ), xs ), ys ), zs ), ws ),
            -kValueBounds, kValueBounds );
    }

public:
    static FS_FORCEINLINE float32v InterpHermiteDerivative( float32v t )
    {
        return float32v( 6 ) * t * ( float32v( 1 ) - t );
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

        float32v xf = x - xs;
        float32v yf = y - ys;
        xs = InterpHermite( xf );
        ys = InterpHermite( yf );

        float32v v00 = GetValueCoord( seed, x0, y0 );
        float32v v10 = GetValueCoord( seed, x1, y0 );
        float32v v01 = GetValueCoord( seed, x0, y1 );
        float32v v11 = GetValueCoord( seed, x1, y1 );

        gradX = InterpHermiteDerivative( xf ) * ( Lerp( v10, v11, ys ) - Lerp( v00, v01, ys ) );
        gradY = InterpHermiteDerivative( yf ) * ( Lerp( v01, v11, xs ) - Lerp( v00, v10, xs ) );

        return Lerp( Lerp( v00, v10, xs ), Lerp( v01, v11, xs ), ys );
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

        float32v xf = x - xs;
        float32v yf = y - ys;
        float32v zf = z - zs;
        xs = InterpHermite( xf );
        ys = InterpHermite( yf );
        zs = InterpHermite( zf );

        float32v v000 = GetValueCoord( seed, x0, y0, z0 );
        float32v v100 = GetValueCoord( seed, x1, y0, z0 );
        float32v v010 = GetValueCoord( seed, x0, y1, z0 );
        float32v v110 = GetValueCoord( seed, x1, y1, z0 );
        float32v v001 = GetValueCoord( seed, x0, y0, z1 );
        float32v v101 = GetValueCoord( seed, x1, y0, z1 );
        float32v v011 = GetValueCoord( seed, x0, y1, z1 );
        float32v v111 = GetValueCoord( seed, x1, y1, z1 );

        gradX = InterpHermiteDerivative( xf ) *
            ( Lerp( Lerp( v100, v110, ys ), Lerp( v101, v111, ys ), zs ) - Lerp( Lerp( v000, v010, ys ), Lerp( v001, v011, ys ), zs ) );
        gradY = InterpHermiteDerivative( yf ) *
            ( Lerp( Lerp( v010, v110, xs ), Lerp( v011, v111, xs ), zs ) - Lerp( Lerp( v000, v100, xs ), Lerp( v001, v101, xs ), zs ) );
        gradZ = InterpHermiteDerivative( zf ) *
            ( Lerp( Lerp( v001, v101, xs ), Lerp( v011, v111, xs ), ys ) - Lerp( Lerp( v000, v100, xs ), Lerp( v010, v110, xs ), ys ) );

        return Lerp( Lerp(
                         Lerp( v000, v100, xs ),
                         Lerp( v010, v110, xs ), ys ),
                     Lerp(
                         Lerp( v001, v101, xs ),
                         Lerp( v011, v111, xs ), ys ),
                     zs );
    }

    bool FS_VECTORCALL GenGradient( int32v seed, float32v x, float32v y, float32v& value, float32v& dx, float32v& dy ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y );

        value = ValueGradient( seed, x, y, dx, dy );

        float32v slope = this->ScaleOutputSlope( -kValueBounds, kValueBounds );
        dx *= slope;
        dy *= slope;
        value = this->ScaleOutput( value, -kValueBounds, kValueBounds );
        return true;
    }

    bool FS_VECTORCALL GenGradient( int32v seed, float32v x, float32v y, float32v z, float32v& value, float32v& dx, float32v& dy, float32v& dz ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y, z );

        value = ValueGradient( seed, x, y, z, dx, dy, dz );

        float32v slope = this->ScaleOutputSlope( -kValueBounds, kValueBounds );
        dx *= slope;
        dy *= slope;
        dz *= slope;
        value = this->ScaleOutput( value, -kValueBounds, kValueBounds );
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

        float32v xf = x - xs;
        float32v yf = y - ys;
        float32v zf = z - zs;
        float32v wf = w - ws;
        float32v derivative[4] = { InterpHermiteDerivative( xf ), InterpHermiteDerivative( yf ), InterpHermiteDerivative( zf ), InterpHermiteDerivative( wf ) };
        xs = InterpHermite( xf );
        ys = InterpHermite( yf );
        zs = InterpHermite( zf );
        ws = InterpHermite( wf );

        float32v corner[16];
        for( int i = 0; i < 16; i++ )
        {
            corner[i] = GetValueCoord( seed, i & 1 ? x1 : x0, i & 2 ? y1 : y0, i & 4 ? z1 : z0, i & 8 ? w1 : w0 );
        }

        float32v* gradient[4] = { &gradX, &gradY, &gradZ, &gradW };
        for( int axis = 0; axis < 4; axis++ )
        {
            float32v across[16];
            for( int i = 0; i < 16; i++ )
            {
                across[i] = corner[i | ( 1 << axis )] - corner[i & ~( 1 << axis )];
            }
            *gradient[axis] = derivative[axis] * Lerp4D( across, xs, ys, zs, ws );
        }

        return Lerp4D( corner, xs, ys, zs, ws );
    }

    bool FS_VECTORCALL GenGradient( int32v seed, float32v x, float32v y, float32v z, float32v w, float32v& value, float32v& dx, float32v& dy, float32v& dz, float32v& dw ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y, z, w );

        value = ValueGradient( seed, x, y, z, w, dx, dy, dz, dw );

        float32v slope = this->ScaleOutputSlope( -kValueBounds, kValueBounds );
        dx *= slope;
        dy *= slope;
        dz *= slope;
        dw *= slope;
        value = this->ScaleOutput( value, -kValueBounds, kValueBounds );
        return true;
    }
};
