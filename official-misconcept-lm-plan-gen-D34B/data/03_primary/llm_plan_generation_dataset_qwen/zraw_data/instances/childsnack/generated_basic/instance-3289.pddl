; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 627027

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child318 child92 - child
    bread424 bread249 - bread-portion
    content490 content115 - content-portion
    tray304 tray91 tray96 tray394 - tray
    table65 table47 table44 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray304 kitchen)
     (at tray91 kitchen)
     (at tray96 kitchen)
     (at tray394 kitchen)
     (at_kitchen_bread bread424)
     (at_kitchen_bread bread249)
     (at_kitchen_content content490)
     (at_kitchen_content content115)
     (no_gluten_bread bread424)
     (no_gluten_content content115)
     (allergic_gluten child318)
     (not_allergic_gluten child92)
     (waiting child318 table44)
     (waiting child92 table65)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child318)
     (served child92)
    )
  )
)
