; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 319969

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child449 - child
    bread249 - bread-portion
    content67 - content-portion
    tray112 tray241 tray11 - tray
    table84 table165 table258 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray112 kitchen)
     (at tray241 kitchen)
     (at tray11 kitchen)
     (at_kitchen_bread bread249)
     (at_kitchen_content content67)
     (not_allergic_gluten child449)
     (waiting child449 table84)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child449)
    )
  )
)
