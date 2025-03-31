; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 903881

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child315 child485 - child
    bread450 bread57 - bread-portion
    content322 content173 - content-portion
    tray217 tray351 tray283 tray386 - tray
    table497 table448 table296 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray217 kitchen)
     (at tray351 kitchen)
     (at tray283 kitchen)
     (at tray386 kitchen)
     (at_kitchen_bread bread450)
     (at_kitchen_bread bread57)
     (at_kitchen_content content322)
     (at_kitchen_content content173)
     (no_gluten_bread bread450)
     (no_gluten_content content322)
     (allergic_gluten child315)
     (not_allergic_gluten child485)
     (waiting child315 table448)
     (waiting child485 table296)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child315)
     (served child485)
    )
  )
)
