; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 350243

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child94 child253 - child
    bread435 bread161 - bread-portion
    content42 content440 - content-portion
    tray351 tray262 - tray
    table352 table489 table389 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray351 kitchen)
     (at tray262 kitchen)
     (at_kitchen_bread bread435)
     (at_kitchen_bread bread161)
     (at_kitchen_content content42)
     (at_kitchen_content content440)
     (not_allergic_gluten child253)
     (not_allergic_gluten child94)
     (waiting child94 table489)
     (waiting child253 table489)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child94)
     (served child253)
    )
  )
)
