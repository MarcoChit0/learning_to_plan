; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 726072

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child182 child484 - child
    bread200 bread301 - bread-portion
    content344 content491 - content-portion
    tray202 - tray
    table369 table214 table408 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray202 kitchen)
     (at_kitchen_bread bread200)
     (at_kitchen_bread bread301)
     (at_kitchen_content content344)
     (at_kitchen_content content491)
     (no_gluten_bread bread200)
     (no_gluten_content content491)
     (allergic_gluten child484)
     (not_allergic_gluten child182)
     (waiting child182 table408)
     (waiting child484 table369)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child182)
     (served child484)
    )
  )
)
