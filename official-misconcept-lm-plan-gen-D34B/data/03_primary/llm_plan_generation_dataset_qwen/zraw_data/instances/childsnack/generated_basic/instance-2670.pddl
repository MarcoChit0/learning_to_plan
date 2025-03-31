; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 984125

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child277 - child
    bread425 - bread-portion
    content382 - content-portion
    tray198 tray492 tray419 tray64 - tray
    table499 table176 table327 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray198 kitchen)
     (at tray492 kitchen)
     (at tray419 kitchen)
     (at tray64 kitchen)
     (at_kitchen_bread bread425)
     (at_kitchen_content content382)
     (not_allergic_gluten child277)
     (waiting child277 table176)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child277)
    )
  )
)
