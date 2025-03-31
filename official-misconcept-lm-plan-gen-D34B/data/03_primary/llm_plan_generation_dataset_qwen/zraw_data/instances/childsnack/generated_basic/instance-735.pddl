; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 953856

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child23 child159 - child
    bread33 bread310 - bread-portion
    content473 content215 - content-portion
    tray287 tray394 - tray
    table170 table287 table407 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray287 kitchen)
     (at tray394 kitchen)
     (at_kitchen_bread bread33)
     (at_kitchen_bread bread310)
     (at_kitchen_content content473)
     (at_kitchen_content content215)
     (no_gluten_bread bread33)
     (no_gluten_content content473)
     (allergic_gluten child159)
     (not_allergic_gluten child23)
     (waiting child23 table287)
     (waiting child159 table407)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child23)
     (served child159)
    )
  )
)
