; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 616196

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child416 - child
    bread349 - bread-portion
    content380 - content-portion
    tray310 tray384 tray497 tray1 - tray
    table45 table446 table188 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray310 kitchen)
     (at tray384 kitchen)
     (at tray497 kitchen)
     (at tray1 kitchen)
     (at_kitchen_bread bread349)
     (at_kitchen_content content380)
     (not_allergic_gluten child416)
     (waiting child416 table45)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child416)
    )
  )
)
