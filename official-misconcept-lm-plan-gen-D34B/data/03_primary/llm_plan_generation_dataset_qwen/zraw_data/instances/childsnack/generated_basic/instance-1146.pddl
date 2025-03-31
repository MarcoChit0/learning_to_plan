; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 176139

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child217 child187 - child
    bread375 bread283 - bread-portion
    content307 content23 - content-portion
    tray206 tray219 tray177 - tray
    table336 table43 table107 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray206 kitchen)
     (at tray219 kitchen)
     (at tray177 kitchen)
     (at_kitchen_bread bread375)
     (at_kitchen_bread bread283)
     (at_kitchen_content content307)
     (at_kitchen_content content23)
     (no_gluten_bread bread283)
     (no_gluten_content content307)
     (allergic_gluten child187)
     (not_allergic_gluten child217)
     (waiting child217 table107)
     (waiting child187 table336)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child217)
     (served child187)
    )
  )
)
