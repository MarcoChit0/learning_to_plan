; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 705209

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child372 child256 - child
    bread218 bread343 - bread-portion
    content113 content47 - content-portion
    tray187 tray168 - tray
    table198 table262 table108 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray187 kitchen)
     (at tray168 kitchen)
     (at_kitchen_bread bread218)
     (at_kitchen_bread bread343)
     (at_kitchen_content content113)
     (at_kitchen_content content47)
     (no_gluten_bread bread343)
     (no_gluten_content content113)
     (allergic_gluten child372)
     (not_allergic_gluten child256)
     (waiting child372 table108)
     (waiting child256 table108)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child372)
     (served child256)
    )
  )
)
