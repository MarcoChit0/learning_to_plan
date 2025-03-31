; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 876654

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child93 child98 child477 - child
    bread112 bread1 bread181 - bread-portion
    content389 content281 content323 - content-portion
    tray309 tray132 tray285 tray112 - tray
    table12 table137 table302 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray309 kitchen)
     (at tray132 kitchen)
     (at tray285 kitchen)
     (at tray112 kitchen)
     (at_kitchen_bread bread112)
     (at_kitchen_bread bread1)
     (at_kitchen_bread bread181)
     (at_kitchen_content content389)
     (at_kitchen_content content281)
     (at_kitchen_content content323)
     (not_allergic_gluten child93)
     (not_allergic_gluten child98)
     (not_allergic_gluten child477)
     (waiting child93 table137)
     (waiting child98 table12)
     (waiting child477 table137)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child93)
     (served child98)
     (served child477)
    )
  )
)
