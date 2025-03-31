; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 556575

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child21 child421 - child
    bread273 bread365 - bread-portion
    content287 content325 - content-portion
    tray306 tray412 tray448 - tray
    table418 table99 table475 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray306 kitchen)
     (at tray412 kitchen)
     (at tray448 kitchen)
     (at_kitchen_bread bread273)
     (at_kitchen_bread bread365)
     (at_kitchen_content content287)
     (at_kitchen_content content325)
     (not_allergic_gluten child21)
     (not_allergic_gluten child421)
     (waiting child21 table475)
     (waiting child421 table418)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child21)
     (served child421)
    )
  )
)
