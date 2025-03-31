; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 369161

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child255 child356 - child
    bread35 bread336 - bread-portion
    content30 content395 - content-portion
    tray384 tray268 tray107 tray323 - tray
    table81 table92 table41 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray384 kitchen)
     (at tray268 kitchen)
     (at tray107 kitchen)
     (at tray323 kitchen)
     (at_kitchen_bread bread35)
     (at_kitchen_bread bread336)
     (at_kitchen_content content30)
     (at_kitchen_content content395)
     (not_allergic_gluten child255)
     (not_allergic_gluten child356)
     (waiting child255 table41)
     (waiting child356 table41)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child255)
     (served child356)
    )
  )
)
