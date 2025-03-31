; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 632924

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child270 - child
    bread185 - bread-portion
    content420 - content-portion
    tray305 tray393 tray215 tray148 - tray
    table60 table29 table241 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray305 kitchen)
     (at tray393 kitchen)
     (at tray215 kitchen)
     (at tray148 kitchen)
     (at_kitchen_bread bread185)
     (at_kitchen_content content420)
     (not_allergic_gluten child270)
     (waiting child270 table29)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child270)
    )
  )
)
