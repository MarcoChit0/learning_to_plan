; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 42039

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child435 - child
    bread42 - bread-portion
    content395 - content-portion
    tray332 tray337 tray35 - tray
    table197 table23 table449 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray332 kitchen)
     (at tray337 kitchen)
     (at tray35 kitchen)
     (at_kitchen_bread bread42)
     (at_kitchen_content content395)
     (not_allergic_gluten child435)
     (waiting child435 table449)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child435)
    )
  )
)
