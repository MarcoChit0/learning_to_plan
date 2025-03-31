; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 117197

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child298 - child
    bread14 - bread-portion
    content36 - content-portion
    tray370 tray437 tray107 - tray
    table334 table140 table47 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray370 kitchen)
     (at tray437 kitchen)
     (at tray107 kitchen)
     (at_kitchen_bread bread14)
     (at_kitchen_content content36)
     (not_allergic_gluten child298)
     (waiting child298 table47)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child298)
    )
  )
)
