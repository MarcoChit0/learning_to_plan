; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 934094

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child442 child195 - child
    bread219 bread465 - bread-portion
    content301 content152 - content-portion
    tray157 - tray
    table218 table276 table123 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray157 kitchen)
     (at_kitchen_bread bread219)
     (at_kitchen_bread bread465)
     (at_kitchen_content content301)
     (at_kitchen_content content152)
     (no_gluten_bread bread219)
     (no_gluten_content content301)
     (allergic_gluten child195)
     (not_allergic_gluten child442)
     (waiting child442 table276)
     (waiting child195 table218)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child442)
     (served child195)
    )
  )
)
