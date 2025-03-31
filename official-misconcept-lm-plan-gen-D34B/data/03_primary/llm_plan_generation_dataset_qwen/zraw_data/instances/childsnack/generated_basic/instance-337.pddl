; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 385873

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child328 - child
    bread221 - bread-portion
    content129 - content-portion
    tray261 tray139 tray172 tray396 - tray
    table323 table34 table177 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray261 kitchen)
     (at tray139 kitchen)
     (at tray172 kitchen)
     (at tray396 kitchen)
     (at_kitchen_bread bread221)
     (at_kitchen_content content129)
     (not_allergic_gluten child328)
     (waiting child328 table34)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child328)
    )
  )
)
