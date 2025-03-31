; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 666012

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child253 child489 child33 - child
    bread413 bread444 bread46 - bread-portion
    content340 content344 content48 - content-portion
    tray95 - tray
    table265 table260 table16 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray95 kitchen)
     (at_kitchen_bread bread413)
     (at_kitchen_bread bread444)
     (at_kitchen_bread bread46)
     (at_kitchen_content content340)
     (at_kitchen_content content344)
     (at_kitchen_content content48)
     (not_allergic_gluten child489)
     (not_allergic_gluten child253)
     (not_allergic_gluten child33)
     (waiting child253 table260)
     (waiting child489 table260)
     (waiting child33 table265)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child253)
     (served child489)
     (served child33)
    )
  )
)
