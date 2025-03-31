; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 651251

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child126 - child
    bread33 - bread-portion
    content446 - content-portion
    tray483 - tray
    table481 table386 table239 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray483 kitchen)
     (at_kitchen_bread bread33)
     (at_kitchen_content content446)
     (not_allergic_gluten child126)
     (waiting child126 table239)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child126)
    )
  )
)
