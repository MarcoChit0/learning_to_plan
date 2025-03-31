; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 163491

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child349 child495 - child
    bread412 bread383 - bread-portion
    content478 content385 - content-portion
    tray133 tray26 - tray
    table480 table442 table398 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray133 kitchen)
     (at tray26 kitchen)
     (at_kitchen_bread bread412)
     (at_kitchen_bread bread383)
     (at_kitchen_content content478)
     (at_kitchen_content content385)
     (not_allergic_gluten child495)
     (not_allergic_gluten child349)
     (waiting child349 table442)
     (waiting child495 table480)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child349)
     (served child495)
    )
  )
)
