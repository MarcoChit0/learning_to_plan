; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 225201

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child259 child274 - child
    bread139 bread331 - bread-portion
    content292 content407 - content-portion
    tray347 tray172 - tray
    table336 table92 table3 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray347 kitchen)
     (at tray172 kitchen)
     (at_kitchen_bread bread139)
     (at_kitchen_bread bread331)
     (at_kitchen_content content292)
     (at_kitchen_content content407)
     (no_gluten_bread bread331)
     (no_gluten_content content407)
     (allergic_gluten child274)
     (not_allergic_gluten child259)
     (waiting child259 table336)
     (waiting child274 table336)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child259)
     (served child274)
    )
  )
)
