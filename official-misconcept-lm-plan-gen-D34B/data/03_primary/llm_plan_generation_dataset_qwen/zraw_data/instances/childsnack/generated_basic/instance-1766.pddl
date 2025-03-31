; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 579449

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child53 - child
    bread42 - bread-portion
    content239 - content-portion
    tray284 - tray
    table326 table212 table399 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray284 kitchen)
     (at_kitchen_bread bread42)
     (at_kitchen_content content239)
     (not_allergic_gluten child53)
     (waiting child53 table399)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child53)
    )
  )
)
