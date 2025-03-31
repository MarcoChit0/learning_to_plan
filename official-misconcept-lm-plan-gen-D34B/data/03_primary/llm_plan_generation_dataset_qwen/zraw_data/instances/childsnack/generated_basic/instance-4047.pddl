; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 16571

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child113 child478 child152 - child
    bread462 bread430 bread269 - bread-portion
    content408 content386 content203 - content-portion
    tray90 - tray
    table79 table123 table167 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray90 kitchen)
     (at_kitchen_bread bread462)
     (at_kitchen_bread bread430)
     (at_kitchen_bread bread269)
     (at_kitchen_content content408)
     (at_kitchen_content content386)
     (at_kitchen_content content203)
     (not_allergic_gluten child113)
     (not_allergic_gluten child152)
     (not_allergic_gluten child478)
     (waiting child113 table167)
     (waiting child478 table123)
     (waiting child152 table123)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child113)
     (served child478)
     (served child152)
    )
  )
)
