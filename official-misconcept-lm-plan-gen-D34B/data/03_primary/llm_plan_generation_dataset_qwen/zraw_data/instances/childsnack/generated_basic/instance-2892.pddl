; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 417568

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child191 child339 - child
    bread353 bread257 - bread-portion
    content369 content55 - content-portion
    tray396 tray181 tray320 - tray
    table333 table348 table325 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray396 kitchen)
     (at tray181 kitchen)
     (at tray320 kitchen)
     (at_kitchen_bread bread353)
     (at_kitchen_bread bread257)
     (at_kitchen_content content369)
     (at_kitchen_content content55)
     (no_gluten_bread bread353)
     (no_gluten_content content55)
     (allergic_gluten child339)
     (not_allergic_gluten child191)
     (waiting child191 table348)
     (waiting child339 table333)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child191)
     (served child339)
    )
  )
)
