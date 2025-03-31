; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 74430

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child343 child164 - child
    bread387 bread197 - bread-portion
    content220 content414 - content-portion
    tray261 tray358 - tray
    table27 table126 table493 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray261 kitchen)
     (at tray358 kitchen)
     (at_kitchen_bread bread387)
     (at_kitchen_bread bread197)
     (at_kitchen_content content220)
     (at_kitchen_content content414)
     (no_gluten_bread bread197)
     (no_gluten_content content220)
     (allergic_gluten child164)
     (not_allergic_gluten child343)
     (waiting child343 table27)
     (waiting child164 table126)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child343)
     (served child164)
    )
  )
)
