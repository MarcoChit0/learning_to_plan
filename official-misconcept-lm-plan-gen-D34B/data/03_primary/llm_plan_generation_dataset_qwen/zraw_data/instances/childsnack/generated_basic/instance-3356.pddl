; child-snack task with 2 children and 0.6 gluten factor 
; constant factor of 1.3
; random seed: 569909

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child296 child388 - child
    bread384 bread368 - bread-portion
    content113 content319 - content-portion
    tray345 tray97 - tray
    table168 table160 table13 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray345 kitchen)
     (at tray97 kitchen)
     (at_kitchen_bread bread384)
     (at_kitchen_bread bread368)
     (at_kitchen_content content113)
     (at_kitchen_content content319)
     (no_gluten_bread bread384)
     (no_gluten_content content113)
     (allergic_gluten child296)
     (not_allergic_gluten child388)
     (waiting child296 table13)
     (waiting child388 table160)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child296)
     (served child388)
    )
  )
)
