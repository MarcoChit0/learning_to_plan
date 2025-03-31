; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 961878

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child209 - child
    bread239 - bread-portion
    content384 - content-portion
    tray139 tray384 tray487 tray250 - tray
    table404 table484 table249 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray139 kitchen)
     (at tray384 kitchen)
     (at tray487 kitchen)
     (at tray250 kitchen)
     (at_kitchen_bread bread239)
     (at_kitchen_content content384)
     (not_allergic_gluten child209)
     (waiting child209 table249)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child209)
    )
  )
)
