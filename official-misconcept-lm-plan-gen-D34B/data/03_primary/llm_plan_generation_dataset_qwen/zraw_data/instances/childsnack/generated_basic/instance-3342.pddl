; child-snack task with 2 children and 0.71 gluten factor 
; constant factor of 1.7
; random seed: 988193

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child25 child416 - child
    bread101 bread140 - bread-portion
    content317 content275 - content-portion
    tray174 tray346 tray46 tray227 - tray
    table129 table405 table322 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray174 kitchen)
     (at tray346 kitchen)
     (at tray46 kitchen)
     (at tray227 kitchen)
     (at_kitchen_bread bread101)
     (at_kitchen_bread bread140)
     (at_kitchen_content content317)
     (at_kitchen_content content275)
     (no_gluten_bread bread140)
     (no_gluten_content content317)
     (allergic_gluten child416)
     (not_allergic_gluten child25)
     (waiting child25 table405)
     (waiting child416 table129)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child25)
     (served child416)
    )
  )
)
