; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 245000

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child155 - child
    bread185 - bread-portion
    content160 - content-portion
    tray54 tray366 tray393 - tray
    table274 table392 table175 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray54 kitchen)
     (at tray366 kitchen)
     (at tray393 kitchen)
     (at_kitchen_bread bread185)
     (at_kitchen_content content160)
     (not_allergic_gluten child155)
     (waiting child155 table175)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child155)
    )
  )
)
