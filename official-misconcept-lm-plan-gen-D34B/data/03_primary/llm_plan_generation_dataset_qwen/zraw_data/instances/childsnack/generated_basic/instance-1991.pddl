; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 70550

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child339 child129 - child
    bread59 bread316 - bread-portion
    content367 content136 - content-portion
    tray151 - tray
    table144 table308 table11 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray151 kitchen)
     (at_kitchen_bread bread59)
     (at_kitchen_bread bread316)
     (at_kitchen_content content367)
     (at_kitchen_content content136)
     (no_gluten_bread bread316)
     (no_gluten_content content136)
     (allergic_gluten child339)
     (not_allergic_gluten child129)
     (waiting child339 table308)
     (waiting child129 table308)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child339)
     (served child129)
    )
  )
)
