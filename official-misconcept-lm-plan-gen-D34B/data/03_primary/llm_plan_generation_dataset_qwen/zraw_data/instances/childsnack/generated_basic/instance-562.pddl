; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 36467

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child140 child482 - child
    bread348 bread465 - bread-portion
    content101 content225 - content-portion
    tray339 - tray
    table440 table289 table224 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray339 kitchen)
     (at_kitchen_bread bread348)
     (at_kitchen_bread bread465)
     (at_kitchen_content content101)
     (at_kitchen_content content225)
     (no_gluten_bread bread465)
     (no_gluten_content content225)
     (allergic_gluten child140)
     (not_allergic_gluten child482)
     (waiting child140 table440)
     (waiting child482 table224)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child140)
     (served child482)
    )
  )
)
