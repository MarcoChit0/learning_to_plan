; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 615016

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child410 child482 - child
    bread17 bread477 - bread-portion
    content102 content82 - content-portion
    tray67 - tray
    table203 table67 table456 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray67 kitchen)
     (at_kitchen_bread bread17)
     (at_kitchen_bread bread477)
     (at_kitchen_content content102)
     (at_kitchen_content content82)
     (no_gluten_bread bread17)
     (no_gluten_content content82)
     (allergic_gluten child410)
     (not_allergic_gluten child482)
     (waiting child410 table203)
     (waiting child482 table203)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child410)
     (served child482)
    )
  )
)
