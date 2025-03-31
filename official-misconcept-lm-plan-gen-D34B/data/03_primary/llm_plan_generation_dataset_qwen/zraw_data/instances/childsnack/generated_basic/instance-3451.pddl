; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 945357

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child117 - child
    bread232 - bread-portion
    content396 - content-portion
    tray463 tray334 tray446 tray421 - tray
    table415 table383 table205 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray463 kitchen)
     (at tray334 kitchen)
     (at tray446 kitchen)
     (at tray421 kitchen)
     (at_kitchen_bread bread232)
     (at_kitchen_content content396)
     (not_allergic_gluten child117)
     (waiting child117 table383)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child117)
    )
  )
)
