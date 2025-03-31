; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 563589

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child397 child385 - child
    bread257 bread128 - bread-portion
    content457 content443 - content-portion
    tray477 tray465 tray376 - tray
    table61 table333 table70 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray477 kitchen)
     (at tray465 kitchen)
     (at tray376 kitchen)
     (at_kitchen_bread bread257)
     (at_kitchen_bread bread128)
     (at_kitchen_content content457)
     (at_kitchen_content content443)
     (not_allergic_gluten child397)
     (not_allergic_gluten child385)
     (waiting child397 table333)
     (waiting child385 table333)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child397)
     (served child385)
    )
  )
)
