; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 23252

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child86 child18 - child
    bread249 bread75 - bread-portion
    content125 content169 - content-portion
    tray117 - tray
    table204 table343 table164 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray117 kitchen)
     (at_kitchen_bread bread249)
     (at_kitchen_bread bread75)
     (at_kitchen_content content125)
     (at_kitchen_content content169)
     (not_allergic_gluten child18)
     (not_allergic_gluten child86)
     (waiting child86 table343)
     (waiting child18 table343)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child86)
     (served child18)
    )
  )
)
