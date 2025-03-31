; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 926893

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child251 - child
    bread296 - bread-portion
    content486 - content-portion
    tray148 - tray
    table402 table100 table126 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray148 kitchen)
     (at_kitchen_bread bread296)
     (at_kitchen_content content486)
     (not_allergic_gluten child251)
     (waiting child251 table126)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child251)
    )
  )
)
