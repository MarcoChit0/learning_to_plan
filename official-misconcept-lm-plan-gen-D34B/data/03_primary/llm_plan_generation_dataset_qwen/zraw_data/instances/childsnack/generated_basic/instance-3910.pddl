; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 513799

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child218 - child
    bread23 - bread-portion
    content414 - content-portion
    tray240 tray374 - tray
    table406 table279 table82 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray240 kitchen)
     (at tray374 kitchen)
     (at_kitchen_bread bread23)
     (at_kitchen_content content414)
     (not_allergic_gluten child218)
     (waiting child218 table406)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child218)
    )
  )
)
