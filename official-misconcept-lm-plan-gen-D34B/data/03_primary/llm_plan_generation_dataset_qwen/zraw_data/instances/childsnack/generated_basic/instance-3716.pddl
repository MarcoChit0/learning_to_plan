; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 553544

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child429 - child
    bread356 - bread-portion
    content74 - content-portion
    tray357 tray183 tray215 - tray
    table170 table134 table412 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray357 kitchen)
     (at tray183 kitchen)
     (at tray215 kitchen)
     (at_kitchen_bread bread356)
     (at_kitchen_content content74)
     (not_allergic_gluten child429)
     (waiting child429 table134)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child429)
    )
  )
)
