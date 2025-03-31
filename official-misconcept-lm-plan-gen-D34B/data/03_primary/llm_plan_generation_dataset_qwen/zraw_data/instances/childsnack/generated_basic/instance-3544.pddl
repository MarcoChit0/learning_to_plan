; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 512637

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child54 child371 - child
    bread194 bread334 - bread-portion
    content216 content132 - content-portion
    tray364 tray495 - tray
    table249 table274 table100 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray364 kitchen)
     (at tray495 kitchen)
     (at_kitchen_bread bread194)
     (at_kitchen_bread bread334)
     (at_kitchen_content content216)
     (at_kitchen_content content132)
     (not_allergic_gluten child54)
     (not_allergic_gluten child371)
     (waiting child54 table100)
     (waiting child371 table274)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child54)
     (served child371)
    )
  )
)
