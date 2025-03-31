; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 28328

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child224 - child
    bread308 - bread-portion
    content193 - content-portion
    tray329 - tray
    table202 table311 table298 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray329 kitchen)
     (at_kitchen_bread bread308)
     (at_kitchen_content content193)
     (not_allergic_gluten child224)
     (waiting child224 table311)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child224)
    )
  )
)
