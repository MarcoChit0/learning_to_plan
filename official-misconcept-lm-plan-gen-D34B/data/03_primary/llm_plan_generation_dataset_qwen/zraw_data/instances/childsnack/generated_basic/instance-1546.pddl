; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 954790

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child227 child93 - child
    bread241 bread144 - bread-portion
    content448 content436 - content-portion
    tray43 tray208 tray259 tray48 - tray
    table227 table59 table251 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray43 kitchen)
     (at tray208 kitchen)
     (at tray259 kitchen)
     (at tray48 kitchen)
     (at_kitchen_bread bread241)
     (at_kitchen_bread bread144)
     (at_kitchen_content content448)
     (at_kitchen_content content436)
     (not_allergic_gluten child93)
     (not_allergic_gluten child227)
     (waiting child227 table59)
     (waiting child93 table251)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child227)
     (served child93)
    )
  )
)
