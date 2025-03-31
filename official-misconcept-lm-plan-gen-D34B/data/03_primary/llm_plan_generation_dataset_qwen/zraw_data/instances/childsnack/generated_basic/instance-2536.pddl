; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 15338

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child302 - child
    bread458 - bread-portion
    content435 - content-portion
    tray53 tray420 tray360 - tray
    table130 table37 table246 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray53 kitchen)
     (at tray420 kitchen)
     (at tray360 kitchen)
     (at_kitchen_bread bread458)
     (at_kitchen_content content435)
     (not_allergic_gluten child302)
     (waiting child302 table246)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child302)
    )
  )
)
