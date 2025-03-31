; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 49017

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child155 child219 - child
    bread38 bread324 - bread-portion
    content232 content80 - content-portion
    tray428 - tray
    table108 table390 table76 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray428 kitchen)
     (at_kitchen_bread bread38)
     (at_kitchen_bread bread324)
     (at_kitchen_content content232)
     (at_kitchen_content content80)
     (no_gluten_bread bread38)
     (no_gluten_content content232)
     (allergic_gluten child219)
     (not_allergic_gluten child155)
     (waiting child155 table76)
     (waiting child219 table76)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child155)
     (served child219)
    )
  )
)
