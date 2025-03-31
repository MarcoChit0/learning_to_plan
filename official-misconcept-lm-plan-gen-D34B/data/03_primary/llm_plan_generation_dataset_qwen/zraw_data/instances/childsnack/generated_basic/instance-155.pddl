; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 526551

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child187 child303 - child
    bread151 bread376 - bread-portion
    content76 content102 - content-portion
    tray463 tray249 tray350 tray100 - tray
    table170 table470 table414 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray463 kitchen)
     (at tray249 kitchen)
     (at tray350 kitchen)
     (at tray100 kitchen)
     (at_kitchen_bread bread151)
     (at_kitchen_bread bread376)
     (at_kitchen_content content76)
     (at_kitchen_content content102)
     (not_allergic_gluten child303)
     (not_allergic_gluten child187)
     (waiting child187 table470)
     (waiting child303 table470)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child187)
     (served child303)
    )
  )
)
