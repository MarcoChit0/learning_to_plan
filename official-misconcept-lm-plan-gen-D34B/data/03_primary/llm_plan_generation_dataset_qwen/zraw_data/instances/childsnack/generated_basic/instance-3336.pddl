; child-snack task with 2 children and 0.71 gluten factor 
; constant factor of 1.7
; random seed: 123178

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child67 child106 - child
    bread3 bread407 - bread-portion
    content59 content271 - content-portion
    tray391 tray211 tray124 tray227 - tray
    table410 table22 table467 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray391 kitchen)
     (at tray211 kitchen)
     (at tray124 kitchen)
     (at tray227 kitchen)
     (at_kitchen_bread bread3)
     (at_kitchen_bread bread407)
     (at_kitchen_content content59)
     (at_kitchen_content content271)
     (no_gluten_bread bread407)
     (no_gluten_content content271)
     (allergic_gluten child67)
     (not_allergic_gluten child106)
     (waiting child67 table22)
     (waiting child106 table467)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child67)
     (served child106)
    )
  )
)
