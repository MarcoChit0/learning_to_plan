; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 868509

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child12 child253 - child
    bread346 bread342 - bread-portion
    content297 content103 - content-portion
    tray198 tray379 tray170 - tray
    table143 table146 table182 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray198 kitchen)
     (at tray379 kitchen)
     (at tray170 kitchen)
     (at_kitchen_bread bread346)
     (at_kitchen_bread bread342)
     (at_kitchen_content content297)
     (at_kitchen_content content103)
     (not_allergic_gluten child12)
     (not_allergic_gluten child253)
     (waiting child12 table146)
     (waiting child253 table143)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child12)
     (served child253)
    )
  )
)
