; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 31882

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child246 child270 - child
    bread418 bread301 - bread-portion
    content10 content428 - content-portion
    tray216 tray320 tray483 tray105 - tray
    table387 table436 table496 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray216 kitchen)
     (at tray320 kitchen)
     (at tray483 kitchen)
     (at tray105 kitchen)
     (at_kitchen_bread bread418)
     (at_kitchen_bread bread301)
     (at_kitchen_content content10)
     (at_kitchen_content content428)
     (no_gluten_bread bread301)
     (no_gluten_content content428)
     (allergic_gluten child270)
     (not_allergic_gluten child246)
     (waiting child246 table496)
     (waiting child270 table387)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child246)
     (served child270)
    )
  )
)
