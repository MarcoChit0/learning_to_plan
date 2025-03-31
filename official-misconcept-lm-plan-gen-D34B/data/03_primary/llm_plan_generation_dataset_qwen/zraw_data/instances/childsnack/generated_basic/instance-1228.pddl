; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 705912

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child349 child453 child210 - child
    bread199 bread495 bread6 - bread-portion
    content266 content385 content249 - content-portion
    tray261 tray58 tray439 - tray
    table82 table133 table425 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray261 kitchen)
     (at tray58 kitchen)
     (at tray439 kitchen)
     (at_kitchen_bread bread199)
     (at_kitchen_bread bread495)
     (at_kitchen_bread bread6)
     (at_kitchen_content content266)
     (at_kitchen_content content385)
     (at_kitchen_content content249)
     (not_allergic_gluten child349)
     (not_allergic_gluten child453)
     (not_allergic_gluten child210)
     (waiting child349 table425)
     (waiting child453 table133)
     (waiting child210 table425)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child349)
     (served child453)
     (served child210)
    )
  )
)
