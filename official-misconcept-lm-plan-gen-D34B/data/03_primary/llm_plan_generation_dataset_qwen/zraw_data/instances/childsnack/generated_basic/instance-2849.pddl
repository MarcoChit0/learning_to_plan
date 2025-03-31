; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 710255

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child333 - child
    bread293 - bread-portion
    content92 - content-portion
    tray128 tray195 - tray
    table374 table99 table265 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray128 kitchen)
     (at tray195 kitchen)
     (at_kitchen_bread bread293)
     (at_kitchen_content content92)
     (not_allergic_gluten child333)
     (waiting child333 table99)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child333)
    )
  )
)
