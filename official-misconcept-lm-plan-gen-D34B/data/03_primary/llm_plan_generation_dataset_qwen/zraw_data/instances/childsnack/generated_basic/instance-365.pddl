; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 582908

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child51 - child
    bread233 - bread-portion
    content224 - content-portion
    tray261 tray28 tray111 tray409 - tray
    table35 table125 table182 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray261 kitchen)
     (at tray28 kitchen)
     (at tray111 kitchen)
     (at tray409 kitchen)
     (at_kitchen_bread bread233)
     (at_kitchen_content content224)
     (not_allergic_gluten child51)
     (waiting child51 table35)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child51)
    )
  )
)
