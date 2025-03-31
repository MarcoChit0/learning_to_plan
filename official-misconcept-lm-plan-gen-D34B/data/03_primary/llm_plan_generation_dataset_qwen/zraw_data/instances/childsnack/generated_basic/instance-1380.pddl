; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 443485

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child350 child90 - child
    bread463 bread203 - bread-portion
    content269 content170 - content-portion
    tray440 - tray
    table385 table488 table248 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray440 kitchen)
     (at_kitchen_bread bread463)
     (at_kitchen_bread bread203)
     (at_kitchen_content content269)
     (at_kitchen_content content170)
     (no_gluten_bread bread203)
     (no_gluten_content content170)
     (allergic_gluten child90)
     (not_allergic_gluten child350)
     (waiting child350 table385)
     (waiting child90 table488)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child350)
     (served child90)
    )
  )
)
