; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 357309

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child185 - child
    bread407 - bread-portion
    content436 - content-portion
    tray357 tray153 tray418 - tray
    table441 table414 table154 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray357 kitchen)
     (at tray153 kitchen)
     (at tray418 kitchen)
     (at_kitchen_bread bread407)
     (at_kitchen_content content436)
     (not_allergic_gluten child185)
     (waiting child185 table441)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child185)
    )
  )
)
