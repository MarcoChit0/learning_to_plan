; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 300330

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child465 - child
    bread364 - bread-portion
    content41 - content-portion
    tray321 tray390 tray439 tray399 - tray
    table64 table484 table71 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray321 kitchen)
     (at tray390 kitchen)
     (at tray439 kitchen)
     (at tray399 kitchen)
     (at_kitchen_bread bread364)
     (at_kitchen_content content41)
     (not_allergic_gluten child465)
     (waiting child465 table484)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child465)
    )
  )
)
